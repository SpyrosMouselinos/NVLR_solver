import torch
import torch.nn as nn
from torch.nn import Module as Module
from bert_modules.bert_modules import BertEncoder
from bert_modules.utils import BertLayerNorm, PositionalEncoding


# class SplitModalEmbedder(Module):
#     def __init__(self, config: dict):
#         super(SplitModalEmbedder, self).__init__()
#         self.config = config
#         self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['embedding_dim'],
#                                                 padding_idx=0)
#         self.color_embeddings = nn.Embedding(config['num_colors'] + 1, config['embedding_dim'], padding_idx=0)
#         self.shape_embeddings = nn.Embedding(config['num_shapes'] + 1, config['embedding_dim'], padding_idx=0)
#         self.material_embeddings = nn.Embedding(config['num_materials'] + 1, config['embedding_dim'], padding_idx=0)
#         self.size_embeddings = nn.Embedding(config['num_sizes'] + 1, config['embedding_dim'], padding_idx=0)
#         self.token_type_embeddings = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
#         self.reproject = nn.Linear(3 + 4 * config['embedding_dim'], config['hidden_dim'])
#         return
#
#     def forward(self,
#                 positions,
#                 types,
#                 object_positions,
#                 object_colors,
#                 object_shapes,
#                 object_materials,
#                 object_sizes,
#                 question):
#         type_embeddings = self.token_type_embeddings(types)
#         otype_embeddings = type_embeddings[:, 0:10]
#         qtype_embeddings = type_embeddings[:, 10:]
#
#         object_mask_ = types.eq(1) * 1.0
#         object_mask = object_mask_[:, :10]
#         object_mask = object_mask.unsqueeze(1).unsqueeze(2)
#
#         question_mask_ = types.eq(2) * 1.0
#         question_mask = question_mask_[:, 10:]
#         question_mask = question_mask.unsqueeze(1).unsqueeze(2)
#
#         mixed_mask = torch.cat([object_mask, question_mask], dim=3)
#
#         questions = self.question_embeddings(question)
#         questions = questions + qtype_embeddings
#         op_proj = object_positions
#         oc_proj = self.color_embeddings(object_colors)
#         os_proj = self.shape_embeddings(object_shapes)
#         om_proj = self.material_embeddings(object_materials)
#         oz_proj = self.size_embeddings(object_sizes)
#         object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj], 2)
#         ore = self.reproject(object_related_embeddings)
#         ore = ore + otype_embeddings
#         return ore, questions, object_mask_, question_mask_, mixed_mask
#
# class SplitModalEmbedderDisentangled(Module):
#     def __init__(self, config: dict):
#         super(SplitModalEmbedderDisentangled, self).__init__()
#         self.config = config
#         self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['embedding_dim'],
#                                                 padding_idx=0)
#         self.position_upscale_projection = nn.Linear(3, config['embedding_dim'])
#         self.color_embeddings = nn.Embedding(config['num_colors'] + 1, config['embedding_dim'], padding_idx=0)
#         self.shape_embeddings = nn.Embedding(config['num_shapes'] + 1, config['embedding_dim'], padding_idx=0)
#         self.material_embeddings = nn.Embedding(config['num_materials'] + 1, config['embedding_dim'], padding_idx=0)
#         self.size_embeddings = nn.Embedding(config['num_sizes'] + 1, config['embedding_dim'], padding_idx=0)
#         self.token_type_embeddings = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
#         return
#
#     def forward(self,
#                 positions,
#                 types,
#                 object_positions,
#                 object_colors,
#                 object_shapes,
#                 object_materials,
#                 object_sizes,
#                 question):
#         type_embeddings = self.token_type_embeddings(types)
#         otype_embeddings = type_embeddings[:, 0:10]
#         qtype_embeddings = type_embeddings[:, 10:]
#
#         object_mask_ = types.eq(1) * 1.0
#         object_mask = object_mask_[:, :10]
#         # Extend mask 5 times before expansion #
#         object_mask = torch.cat([object_mask, object_mask, object_mask, object_mask, object_mask], dim=1)
#         object_mask = object_mask.unsqueeze(1).unsqueeze(2)
#
#         question_mask_ = types.eq(2) * 1.0
#         question_mask = question_mask_[:, 10:]
#         question_mask = question_mask.unsqueeze(1).unsqueeze(2)
#
#         mixed_mask = torch.cat([object_mask, question_mask], dim=3)
#
#         questions = self.question_embeddings(question)
#         questions = questions + qtype_embeddings
#
#         op_proj = self.position_upscale_projection(object_positions) + otype_embeddings
#         oc_proj = self.color_embeddings(object_colors) + otype_embeddings
#         os_proj = self.shape_embeddings(object_shapes) + otype_embeddings
#         om_proj = self.material_embeddings(object_materials) + otype_embeddings
#         oz_proj = self.size_embeddings(object_sizes) + otype_embeddings
#
#         return op_proj, oc_proj, os_proj, om_proj, oz_proj, questions, mixed_mask
#
#
# class MLPClassifierHead(Module):
#     def __init__(self, config: dict, use_log_transform=False, mode='raw'):
#         super(MLPClassifierHead, self).__init__()
#         self.linear_layer_1 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
#         self.linear_layer_2 = nn.Linear(config['hidden_dim'], config['num_output_classes'])
#
#         if mode == 'arg':
#             self.softmax_layer = lambda x: torch.argmax(x, dim=1, keepdim=True)
#         elif mode == 'soft':
#             if use_log_transform:
#                 self.softmax_layer = nn.LogSoftmax(dim=1)
#             else:
#                 self.softmax_layer = nn.Softmax(dim=1)
#         elif mode == 'raw':
#             self.softmax_layer = lambda x: x
#         else:
#             raise NotImplementedError(f"Mode: {mode} not implemented in MLPClassifierHead Module...")
#         return
#
#     def forward(self, input):
#         input = nn.ReLU()(self.linear_layer_1(input))
#         return self.softmax_layer(self.linear_layer_2(input))
#
#
# class QuestionEmbedModel(Module):
#     def __init__(self, config: dict):
#         super(QuestionEmbedModel, self).__init__()
#         self.bidirectional = bool(config['use_bidirectional_encoder'])
#         self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], batch_first=True,
#                             bidirectional=self.bidirectional)
#         # self.reduce = nn.Linear(2 * config['hidden_dim'], config['hidden_dim'])
#
#     def forward(self, question):
#         self.lstm.flatten_parameters()
#         question = torch.flip(question, [1])
#         _, (h, c) = self.lstm(question)
#         h = torch.transpose(h, 1, 0)
#         h = h.reshape(h.size(0), h.size(1) * h.size(2))
#         # h = self.reduce(h)
#         return h

class SplitModalObjectEmbedder(Module):
    def __init__(self, config: dict):
        super(SplitModalObjectEmbedder, self).__init__()
        self.config = config
        self.question_embeddings = nn.Embedding(config['question_vocabulary_size'], config['embedding_dim'],
                                                padding_idx=0)
        self.color_embeddings = nn.Embedding(config['num_colors'] + 1, config['embedding_dim'], padding_idx=0)
        self.shape_embeddings = nn.Embedding(config['num_shapes'] + 1, config['embedding_dim'], padding_idx=0)
        self.material_embeddings = nn.Embedding(config['num_materials'] + 1, config['embedding_dim'], padding_idx=0)
        self.size_embeddings = nn.Embedding(config['num_sizes'] + 1, config['embedding_dim'], padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config['num_token_types'], config['hidden_dim'], padding_idx=0)
        self.reproject = nn.Linear(3 + 4 * config['embedding_dim'], config['hidden_dim'])
        return

    def forward(self,
                positions,
                types,
                object_positions,
                object_colors,
                object_shapes,
                object_materials,
                object_sizes,
                question):
        type_embeddings = self.token_type_embeddings(types)
        otype_embeddings = type_embeddings[:, 0:10]
        qtype_embeddings = type_embeddings[:, 10:]

        object_mask_ = types.eq(1) * 1.0
        object_mask = object_mask_[:, :10]
        object_mask = object_mask.unsqueeze(1).unsqueeze(2)

        question_mask_ = types.eq(2) * 1.0
        question_mask = question_mask_[:, 10:]
        question_mask = question_mask.unsqueeze(1).unsqueeze(2)

        mixed_mask = torch.cat([object_mask, question_mask], dim=3)

        questions = self.question_embeddings(question)
        questions = questions + qtype_embeddings
        op_proj = object_positions
        oc_proj = self.color_embeddings(object_colors)
        os_proj = self.shape_embeddings(object_shapes)
        om_proj = self.material_embeddings(object_materials)
        oz_proj = self.size_embeddings(object_sizes)
        object_related_embeddings = torch.cat([op_proj, oc_proj, os_proj, om_proj, oz_proj], 2)
        ore = self.reproject(object_related_embeddings)
        ore = ore + otype_embeddings
        return ore, questions, object_mask_, question_mask_, mixed_mask


### Test ###
import yaml

config = f'model_config.yaml'
with open(config, 'r') as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)

# class DeltaFormer(Module):
#     def __init__(self, config: dict):
#         super(DeltaFormer, self).__init__()
#         self.mme = MultiModalEmbedder(config)
#         self.be = BertEncoder(config)
#         self.classhead = MLPClassifierHead(config)
#         self.concathead = ConcatClassifierHead(config)
#         self.perhead = PerOutputClassifierHead(config)
#         return
#
#     def forward(self, **kwargs):
#         embeddings, mask, obj_mask = self.mme(**kwargs)
#         out, atts = self.be.forward(embeddings, mask, output_all_encoded_layers=False, output_attention_probs=True)
#         item_output = out[-1][:, 0:10]
#         filtered_item_output = item_output * obj_mask[:, 0:10].unsqueeze(2)
#         answer = self.perhead(filtered_item_output)
#         return answer, atts[-1], None
#
#
# class DeltaSQFormer(Module):
#     def __init__(self, config: dict):
#         super(DeltaSQFormer, self).__init__()
#         self.sme = SplitModalEmbedder(config)
#         self.pos_enc = PositionalEncoding(d_model=config['embedding_dim'], dropout=0.1, max_len=50)
#         self.oln = BertLayerNorm(config['hidden_dim'])
#         self.qln = BertLayerNorm(config['hidden_dim'])
#         self.be = BertEncoder(config)
#         self.classhead = MLPClassifierHead(config)
#         self.avghead = PerOutputClassifierHead(config)
#         return
#
#     def forward(self, **kwargs):
#         object_emb, question_emb, _, _, mixed_mask = self.sme(**kwargs)
#         object_emb = self.oln(object_emb)
#         question_emb = self.pos_enc(question_emb)
#         question_emb = self.qln(question_emb)
#         embeddings = torch.cat([object_emb, question_emb], dim=1)
#         out, atts = self.be.forward(embeddings, mixed_mask, output_all_encoded_layers=False,
#                                     output_attention_probs=True)
#         item_output = out[-1][:, 0]
#         answer = self.classhead(item_output)
#
#         # answer = self.avghead(out[-1])
#         return answer, atts, None
#
#
# class DeltaSQFormerCross(Module):
#     def __init__(self, config: dict):
#         super(DeltaSQFormerCross, self).__init__()
#         self.sme = SplitModalEmbedder(config)
#         self.pos_enc = PositionalEncoding(d_model=config['embedding_dim'], dropout=0.1, max_len=50)
#         self.oln = BertLayerNorm(config['hidden_dim'])
#         self.qln = BertLayerNorm(config['hidden_dim'])
#         self.be = BertEncoder(config)
#         self.classhead = MLPClassifierHead(config)
#         return
#
#     @staticmethod
#     def calc_cross_mask(omask, qmask):
#         stacked_om = torch.stack([omask] * omask.size(1), dim=1)
#         stacked_qm = torch.stack([qmask] * qmask.size(1), dim=1)
#         cross_mask = torch.einsum("bij,bi->bij", stacked_qm, omask) + torch.einsum("bij,bi->bij", stacked_om, qmask)
#         cross_mask = (1.0 - cross_mask) * -10000.0
#         return cross_mask.unsqueeze(1)
#
#     def forward(self, **kwargs):
#         object_emb, question_emb, omask, qmask, _ = self.sme(**kwargs)
#         cross_mask = self.calc_cross_mask(omask, qmask)
#         object_emb = self.oln(object_emb)
#         question_emb = self.pos_enc(question_emb)
#         question_emb = self.qln(question_emb)
#         embeddings = torch.cat([object_emb, question_emb], dim=1)
#         out, atts = self.be.forward(embeddings, cross_mask, output_all_encoded_layers=False,
#                                     output_attention_probs=True)
#         item_output = out[-1][:, 0]
#         answer = self.classhead(item_output)
#
#         return answer, atts, item_output
#
#
# class DeltaQFormer(Module):
#     def __init__(self, config: dict):
#         super(DeltaQFormer, self).__init__()
#         self.qe = QuestionOnlyEmbedder(config)
#         self.pos_enc = PositionalEncoding(d_model=config['embedding_dim'], dropout=0.1, max_len=50)
#         self.be = BertEncoder(config)
#         self.classhead = MLPClassifierHead(config)
#         return
#
#     def forward(self, **kwargs):
#         embeddings, mask = self.qe(**kwargs)
#         embeddings = self.pos_enc(embeddings)
#         out, atts = self.be.forward(embeddings, mask, output_all_encoded_layers=False, output_attention_probs=True)
#         item_output = out[-1][:, 0]
#         answer = self.classhead(item_output)
#         return answer, atts[-1], None
#
#
# class DeltaSQFormerLinear(Module):
#     def __init__(self, config: dict):
#         super(DeltaSQFormerLinear, self).__init__()
#         self.config = config
#         self.smel = SplitModalEmbedderLinear(config)
#         self.pos_enc = PositionalEncoding(d_model=config['embedding_dim'], dropout=0.1, max_len=50)
#         self.oln = BertLayerNorm(config['hidden_dim'])
#         self.qln = BertLayerNorm(config['hidden_dim'])
#         self.sln = BertLayerNorm(config['hidden_dim'])
#         self.be = BertEncoder(config)
#         self.classhead = MLPClassifierHead(config)
#         if 'num_special_heads' in self.config:
#             self.num_special_heads = self.config['num_special_heads']
#         else:
#             self.num_special_heads = 4
#         self.special_heads = nn.Parameter(torch.randn(self.num_special_heads, config['hidden_dim']),
#                                           requires_grad=True)
#         return
#
#     @staticmethod
#     def calc_cross_mask(a, b):
#         stacked_om = torch.stack([a] * a.size(1), dim=1)
#         stacked_qm = torch.stack([b] * b.size(1), dim=1)
#         cross_mask = torch.einsum("bij,bi->bij", stacked_qm, a) + torch.einsum("bij,bi->bij", stacked_om, b)
#         cross_mask = (1.0 - cross_mask) * -10000.0
#         return cross_mask.unsqueeze(1)
#
#     def forward(self, **kwargs):
#         ore, questions, stype_embeddings, mixed_mask, special_mask = self.smel(**kwargs)
#         cross_mask = self.calc_cross_mask(mixed_mask, special_mask)
#         object_emb = self.oln(ore)
#         question_emb = self.pos_enc(questions)
#         question_emb = self.qln(question_emb)
#         special_emb = self.special_heads.repeat((question_emb.size(0), 1, 1))
#         special_emb = special_emb + stype_embeddings
#         special_emb = self.sln(special_emb)
#
#         embeddings = torch.cat([object_emb, question_emb, special_emb], dim=1)
#         out, atts = self.be.forward(embeddings, cross_mask, output_all_encoded_layers=False,
#                                     output_attention_probs=True)
#         item_output = torch.mean(out[-1][:, -self.num_special_heads:], dim=1)
#         answer = self.classhead(item_output)
#
#         return answer, atts, item_output
#
