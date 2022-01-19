import torch
import torch.nn as nn
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
    def __init__(self, config: dict, use_log_transform=False, mode='raw'):
        super(MLPClassifierHead, self).__init__()
        self.linear_layer_1 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.linear_layer_2 = nn.Linear(config['hidden_dim'], config['num_output_classes'])

        if mode == 'arg':
            self.softmax_layer = lambda x: torch.argmax(x, dim=1, keepdim=True)
        elif mode == 'soft':
            if use_log_transform:
                self.softmax_layer = nn.LogSoftmax(dim=1)
            else:
                self.softmax_layer = nn.Softmax(dim=1)
        elif mode == 'raw':
            self.softmax_layer = lambda x: x
        else:
            raise NotImplementedError(f"Mode: {mode} not implemented in MLPClassifierHead Module...")
        return

    def forward(self, input):
        input = nn.ReLU()(self.linear_layer_1(input))
        return self.softmax_layer(self.linear_layer_2(input))


class NVLRformer(Module):
    def __init__(self, config: dict):
        super(NVLRformer, self).__init__()
        self.sme = SplitModalObjectEmbedder(config)
        self.be = BertEncoder(config)
        self.classhead = MLPClassifierHead(config)
        return

    @staticmethod
    def calc_patches(masks, qmasks):
        mask_1 = torch.cat([masks[:, 0], torch.stack([torch.FloatTensor([0] * 16)] * masks.size(0), dim=0)], dim=1)
        mask_2 = torch.cat([torch.stack([torch.FloatTensor([0] * 8)] * masks.size(0), dim=0), masks[:, 1],
                            torch.stack([torch.FloatTensor([0] * 8)] * masks.size(0), dim=0)], dim=1)
        mask_3 = torch.cat([torch.stack([torch.FloatTensor([0] * 16)] * masks.size(0), dim=0), masks[:, 2]], dim=1)
        pad_mask1 = torch.stack([mask_1] * 8, dim=1)
        pad_mask2 = torch.stack([mask_2] * 8, dim=1)
        pad_mask3 = torch.stack([mask_3] * 8, dim=1)
        qmasks_ = torch.stack([qmasks] * 24, dim=1)
        full_mask = torch.cat([pad_mask1, pad_mask2, pad_mask3], dim=1)
        full_mask = torch.cat([full_mask, qmasks_], dim=2)
        # Now fix the resting Q #
        flat_obj_mask = torch.cat([masks.view(-1, 24), qmasks], dim=1)
        flat_obj_mask = torch.stack([flat_obj_mask] * 30, dim=1)

        full_mask = torch.cat([full_mask, flat_obj_mask], dim=1).unsqueeze(1)

        cross_mask = (1.0 - full_mask) * -10000.0
        return cross_mask

    def forward(self, objects, question, mask, qmask, labels):
        final_obj, final_q, final_mask, final_qmask = self.sme(objects, question, mask, qmask)
        cross_mask = self.calc_patches(final_mask, final_qmask)
        embeddings = torch.cat([final_obj, final_q], dim=1)
        out, atts = self.be.forward(embeddings, cross_mask, output_all_encoded_layers=False,
                                    output_attention_probs=True)
        item_output = out[-1][:, 24, :]
        answer = self.classhead(item_output)
        return answer, atts, item_output
