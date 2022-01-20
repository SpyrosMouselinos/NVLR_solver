import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module as Module
from bert_modules.bert_modules import BertEncoder
from bert_modules.utils import BertLayerNorm, PositionalEncoding


class SplitModalObjectEmbedder(Module):
    def __init__(self, config: dict):
        super(SplitModalObjectEmbedder, self).__init__()
        self.config = config
        self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['hidden_dim'],
                                                padding_idx=-1)
        self.pos_enc = PositionalEncoding(d_model=config['hidden_dim'], dropout=0.1,
                                          max_len=config['max_question_tokens_per_scene'])
        self.reproject = nn.Linear(config['n_object_dims'], config['hidden_dim'])
        # Objects / Question / Dummy #
        self.visual_token = nn.Parameter(torch.randn(1, config['hidden_dim']), requires_grad=True)
        self.question_token = nn.Parameter(torch.randn(1, config['hidden_dim']), requires_grad=True)
        self.scene_1_embedding = nn.Parameter(torch.randn(1, config['hidden_dim']), requires_grad=True)
        self.scene_2_embedding = nn.Parameter(torch.randn(1, config['hidden_dim']), requires_grad=True)
        self.scene_3_embedding = nn.Parameter(torch.randn(1, config['hidden_dim']), requires_grad=True)

        # Question LN #
        self.qln = BertLayerNorm(config['hidden_dim'])
        # Scenes LN #
        self.s1ln = BertLayerNorm(config['hidden_dim'])
        self.s2ln = BertLayerNorm(config['hidden_dim'])
        self.s3ln = BertLayerNorm(config['hidden_dim'])
        return

    def forward(self,
                objects, question, mask, qmask):
        # Split items on image slots #
        objects_scene_1 = objects[:, 0, :, :]
        objects_scene_2 = objects[:, 1, :, :]
        objects_scene_3 = objects[:, 2, :, :]
        # Reproject #
        r_objects_scene_1 = self.reproject(objects_scene_1)
        r_objects_scene_2 = self.reproject(objects_scene_2)
        r_objects_scene_3 = self.reproject(objects_scene_3)
        # Add scene embedding #
        se_r_objects_scene_1 = r_objects_scene_1 + self.scene_1_embedding
        se_r_objects_scene_2 = r_objects_scene_2 + self.scene_2_embedding
        se_r_objects_scene_3 = r_objects_scene_3 + self.scene_3_embedding
        # Add Visual Token Mask #
        v_se_r_objects_scene_1 = se_r_objects_scene_1 + self.visual_token
        v_se_r_objects_scene_2 = se_r_objects_scene_2 + self.visual_token
        v_se_r_objects_scene_3 = se_r_objects_scene_3 + self.visual_token
        # Finalize Scenes #
        final_s1 = self.s1ln(v_se_r_objects_scene_1)
        final_s2 = self.s1ln(v_se_r_objects_scene_2)
        final_s3 = self.s1ln(v_se_r_objects_scene_3)

        # Get questions #
        question_embedded = self.question_embeddings(question)
        # Add question embedding #
        q_question_embedded = question_embedded + self.question_token
        # Add positional encoding #
        pos_q_question_embedded = self.pos_enc(q_question_embedded)
        # Finalize question #
        final_q = self.qln(pos_q_question_embedded)

        # Concatenate Object Tokens #
        final_obj = torch.cat([final_s1, final_s2, final_s3], dim=1)
        # Concatenate Mask Tokens #
        return final_obj, final_q, mask, qmask


class MLPClassifierHead(Module):
    def __init__(self, config: dict):
        super(MLPClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2 = nn.Linear(config['hidden_dim'], config['num_output_classes'])

    def forward(self, input):
        relued = F.relu(self.linear_layer_1(input), inplace=True)
        outprob = F.sigmoid(self.linear_layer_2(relued))
        return outprob


class NVLRformer(Module):
    def __init__(self, config: dict):
        super(NVLRformer, self).__init__()
        self.sme = SplitModalObjectEmbedder(config)
        self.be = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        if int(config['mask_type']) == 1:
            self.mask_calculation = self.calc_patches
        else:
            self.mask_calculation = self.simple_mask
        return

    @staticmethod
    def calc_patches(masks, qmasks):
        device = masks.device
        dummy_max  = torch.stack([torch.FloatTensor([0] * 24)] * masks.size(0), dim=0)
        dummy_max  = dummy_max.to(device)
        dummy_full = torch.stack([torch.FloatTensor([0] * 16)] * masks.size(0), dim=0)
        dummy_full = dummy_full.to(device)
        dummy_half = torch.stack([torch.FloatTensor([0] * 8)] * masks.size(0), dim=0)
        dummy_half = dummy_half.to(device)
        mask_1 = torch.cat([masks[:, 0], dummy_full], dim=1)
        mask_2 = torch.cat([dummy_half, masks[:, 1], dummy_half], dim=1)
        mask_3 = torch.cat([dummy_full, masks[:, 2]], dim=1)
        pad_mask1 = torch.stack([mask_1] * 8, dim=1)
        pad_mask2 = torch.stack([mask_2] * 8, dim=1)
        pad_mask3 = torch.stack([mask_3] * 8, dim=1)
        qmasks_ = torch.stack([qmasks] * 24, dim=1)
        full_mask = torch.cat([pad_mask1, pad_mask2, pad_mask3], dim=1)
        full_mask = torch.cat([full_mask, qmasks_], dim=2)
        # Now fix the resting Q #
        flat_obj_mask = torch.cat([dummy_max, qmasks], dim=1)
        flat_obj_mask = torch.stack([flat_obj_mask] * 30, dim=1)

        full_mask = torch.cat([full_mask, flat_obj_mask], dim=1).unsqueeze(1)

        cross_mask = (1.0 - full_mask) * -10000.0
        return cross_mask

    @staticmethod
    def simple_mask(masks, qmasks):
        full_mask = torch.stack([torch.cat([masks.view(-1, 24), qmasks], dim=-1)] * 54, dim=1).unsqueeze(1)
        cross_mask = (1.0 - full_mask) * -10000.0
        return cross_mask

    def forward(self, objects, question, mask, qmask, labels):
        final_obj, final_q, final_mask, final_qmask = self.sme(objects, question, mask, qmask)
        cross_mask = self.simple_mask(final_mask, final_qmask)
        embeddings = torch.cat([final_obj, final_q], dim=1)
        out, atts = self.be.forward(embeddings, cross_mask, output_all_encoded_layers=False,
                                    output_attention_probs=True)
        item_output = torch.mean(out[-1][:, 0:24, :], dim=1)
        answer = self.classhead(item_output)
        return answer, atts, item_output
