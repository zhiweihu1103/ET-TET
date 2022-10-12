import torch
import torch.nn as nn
import trm

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_pretrained_bert.modeling import BertEncoder, BertConfig, BertLayerNorm

class TET(nn.Module):
    def __init__(self, args, num_entities, num_rels, num_types, num_cluster):
        super(TET, self).__init__()
        self.embedding_dim = args['hidden_dim']
        self.embedding_range = 10 / self.embedding_dim
        self.num_rels = num_rels + num_cluster
        self.use_cuda = args['cuda']
        self.dataset = args['dataset']
        self.sample_ent2pair_size = args['sample_ent2pair_size']
        self.tt_ablation = args['tt_ablation']
        self.pooling = args['pair_pooling']
        self.device = torch.device('cuda')
        self.num_nodes = num_entities + num_types

        self.layer = TETLayer(args, self.embedding_dim, num_types, args['temperature'])

        self.entity = nn.Parameter(torch.randn(self.num_nodes, self.embedding_dim))
        nn.init.uniform_(tensor=self.entity, a=-self.embedding_range, b=self.embedding_range)
        self.relation = nn.Parameter(torch.randn(self.num_rels, self.embedding_dim))
        nn.init.uniform_(tensor=self.relation, a=-self.embedding_range, b=self.embedding_range)

        self.bert_nlayer = args['bert_nlayer']
        self.bert_nhead = args['bert_nhead']
        self.bert_ff_dim = args['bert_ff_dim']
        self.bert_activation = args['bert_activation']
        self.bert_hidden_dropout = args['bert_hidden_dropout']
        self.bert_attn_dropout = args['bert_attn_dropout']
        self.local_pos_size = args['local_pos_size']
        self.bert_layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)
        self.local_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.local_cls, std=self.embedding_range)
        self.local_pos_embeds = nn.Embedding(self.local_pos_size, self.embedding_dim)
        torch.nn.init.normal_(self.local_pos_embeds.weight, std=self.embedding_range)
        bert_config = BertConfig(0, hidden_size=self.embedding_dim,
                                 num_hidden_layers=self.bert_nlayer // 2,
                                 num_attention_heads=self.bert_nhead,
                                 intermediate_size=self.bert_ff_dim,
                                 hidden_act=self.bert_activation,
                                 hidden_dropout_prob=self.bert_hidden_dropout,
                                 attention_probs_dropout_prob=self.bert_attn_dropout,
                                 max_position_embeddings=0,
                                 type_vocab_size=0,
                                 initializer_range=self.embedding_range)
        self.bert_encoder = BertEncoder(bert_config)

        self.pair_layer = args['pair_layer']
        self.pair_head = args['pair_head']
        self.pair_dropout = args['pair_dropout']
        self.pair_ff_dim = args['pair_ff_dim']
        self.pair_pos_embeds = nn.Embedding(1 + 2*self.sample_ent2pair_size, self.embedding_dim)
        torch.nn.init.normal_(self.pair_pos_embeds.weight, std=self.embedding_range)
        pair_encoder_layers = TransformerEncoderLayer(self.embedding_dim, self.pair_head, self.pair_ff_dim, self.pair_dropout)
        self.pair_encoder = TransformerEncoder(pair_encoder_layers, self.pair_layer)

    def convert_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

    def forward(self, et_content, kg_content, sample_ent2pair):
        batch_size, et_neighbor_size = et_content[:, :, 2].size()
        et_types = torch.index_select(self.entity, 0, et_content[:, :, 2].view(-1)).view(batch_size, et_neighbor_size, -1)
        et_relations_types = et_content[:, :, 1]
        et_relations = torch.index_select(self.relation, 0, et_relations_types.view(-1) % self.num_rels).view(batch_size, et_neighbor_size, -1)
        et_relations[et_relations_types >= self.num_rels] = et_relations[et_relations_types >= self.num_rels] * -1

        if 'YAGO' in self.dataset:
            # for YAGO dataset, we should use cluster and type context pair to represent the KG relation
            batch_size, kg_neighbor_size = kg_content[:, :, 2].size()
            kg_entities = torch.index_select(self.entity, 0, kg_content[:, :, 2].view(-1)).view(batch_size, kg_neighbor_size, -1)
            _, pair_neighbor_size, _ = sample_ent2pair.size()
            kg_entity2pair = torch.index_select(sample_ent2pair, 0, kg_content[:, :, 2].view(-1)).view(batch_size, kg_neighbor_size, pair_neighbor_size, -1)
            pair_cluster = kg_entity2pair[:, :, :, 0]
            pair_type = kg_entity2pair[:, :, :, 1]
            pair_cluster_embs = torch.index_select(self.relation, 0, pair_cluster.view(-1)).view(-1, pair_neighbor_size, self.embedding_dim)
            pair_type_embs = torch.index_select(self.entity, 0, pair_type.view(-1)).view(-1, pair_neighbor_size, self.embedding_dim)
            kg_relations_types = kg_content[:, :, 1]
            kg_relations = torch.index_select(self.relation, 0, kg_relations_types.view(-1) % self.num_rels).view(-1, 1, self.embedding_dim)
            pairs = torch.cat((pair_cluster_embs, pair_type_embs), 2).view(-1, 2 * pair_cluster_embs.shape[1], pair_cluster_embs.shape[2])
            ent_pairs = torch.cat([kg_relations, pairs], 1).transpose(1, 0)  # [1 + num_pairs, bs, emb_dim]
            ent_pairs_pos = torch.arange(ent_pairs.shape[0], dtype=torch.long, device=self.device).repeat(ent_pairs.shape[1], 1)
            ent_pairs_pos_embeddings = self.pair_pos_embeds(ent_pairs_pos).transpose(1, 0)
            ent_pairs_embs = ent_pairs + ent_pairs_pos_embeddings
            mask = torch.zeros((ent_pairs_embs.shape[1], ent_pairs_embs.shape[0])).bool().to(self.device)
            x = self.pair_encoder(ent_pairs_embs, src_key_padding_mask=mask)

            if self.pooling == 'max':
                x, _ = torch.max(x, dim=0)
            elif self.pooling == "avg":
                x = torch.mean(x, dim=0)
            elif self.pooling == "min":
                x, _ = torch.min(x, dim=0)
            kg_relations = x.view(batch_size, -1, self.embedding_dim)
            kg_relations[kg_relations_types >= self.num_rels] = kg_relations[kg_relations_types >= self.num_rels] * -1
        else:
            batch_size, kg_neighbor_size = kg_content[:, :, 2].size()
            kg_entities = torch.index_select(self.entity, 0, kg_content[:, :, 2].view(-1)).view(batch_size, kg_neighbor_size, -1)
            kg_relations_types = kg_content[:, :, 1]
            kg_relations = torch.index_select(self.relation, 0, kg_relations_types.view(-1) % self.num_rels).view(batch_size, kg_neighbor_size, -1)
            kg_relations[kg_relations_types >= self.num_rels] = kg_relations[kg_relations_types >= self.num_rels] * -1

        et_merge = torch.cat([et_types, et_relations], dim=1).view(-1, 2, self.embedding_dim)
        et_pos = self.local_pos_embeds(torch.arange(0, 3, device=self.device)).unsqueeze(0).repeat(et_merge.shape[0], 1, 1)
        et_merge = torch.cat([self.local_cls.expand(et_merge.size(0), 1, self.embedding_dim), et_merge], dim=1) + et_pos
        et_merge = self.bert_layer_norm(et_merge)
        et_merge = self.bert_encoder(et_merge, self.convert_mask(et_merge.new_ones(et_merge.size(0), et_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1][:, 0].view(batch_size, -1, self.embedding_dim)

        kg_merge = torch.cat([kg_entities, kg_relations], dim=1).view(-1, 2, self.embedding_dim)
        kg_pos = self.local_pos_embeds(torch.arange(0, 3, device=self.device)).unsqueeze(0).repeat(kg_merge.shape[0], 1, 1)
        kg_merge = torch.cat([self.local_cls.expand(kg_merge.size(0), 1, self.embedding_dim), kg_merge], dim=1) + kg_pos
        kg_merge = self.bert_layer_norm(kg_merge)
        kg_merge = self.bert_encoder(kg_merge, self.convert_mask(kg_merge.new_ones(kg_merge.size(0), kg_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1][:, 0].view(batch_size, -1, self.embedding_dim)
        if self.tt_ablation == 'all':
            et_kg_merge = torch.cat([et_types, et_relations, kg_entities, kg_relations], dim=1).view(batch_size, -1, self.embedding_dim)
        elif self.tt_ablation == 'triple':
            et_kg_merge = torch.cat([kg_entities, kg_relations], dim=1).view(batch_size, -1, self.embedding_dim)
        elif self.tt_ablation == 'type':
            et_kg_merge = torch.cat([et_types, et_relations], dim=1).view(batch_size, -1, self.embedding_dim)

        _, et_kg_size, _ = et_kg_merge.size()
        if et_kg_size >= self.local_pos_size-1:
            et_kg_merge = et_kg_merge[:, 0:self.local_pos_size-1, :]
            et_kg_size = self.local_pos_size-1
        et_kg_pos = self.local_pos_embeds(torch.arange(0, et_kg_size + 1, device=self.device)).unsqueeze(0).repeat(et_kg_merge.shape[0], 1, 1)
        et_kg_merge = torch.cat([self.local_cls.expand(et_kg_merge.size(0), 1, self.embedding_dim), et_kg_merge], dim=1) + et_kg_pos
        et_kg_merge = self.bert_layer_norm(et_kg_merge)
        et_kg_merge = self.bert_encoder(et_kg_merge, self.convert_mask(et_kg_merge.new_ones(et_merge.size(0), et_kg_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1][:, 0].view(batch_size, -1, self.embedding_dim)

        if self.tt_ablation == 'all':
            local_embedding = torch.cat([et_merge, kg_merge], dim=1)
        elif self.tt_ablation == 'triple':
            local_embedding = kg_merge
        elif self.tt_ablation == 'type':
            local_embedding = et_merge
        global_embedding = et_kg_merge
        output = self.layer(local_embedding, global_embedding)

        return output


class TETLayer(nn.Module):
    def __init__(self, args, embedding_dim, num_types, temperature):
        super(TETLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(embedding_dim, num_types)
        self.temperature = temperature
        self.device = torch.device('cuda')

        self.trm_nlayer = args['trm_nlayer']
        self.trm_nhead = args['trm_nhead']
        self.trm_hidden_dropout = args['trm_hidden_dropout']
        self.trm_attn_dropout = args['trm_attn_dropout']
        self.trm_ff_dim = args['trm_ff_dim']
        self.global_pos_size = args['global_pos_size']
        self.embedding_range = 10 / self.embedding_dim

        self.global_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.global_cls, std=self.embedding_range)
        self.pos_embeds = nn.Embedding(self.global_pos_size, self.embedding_dim)
        torch.nn.init.normal_(self.pos_embeds.weight, std=self.embedding_range)
        self.layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)

        self.transformer_encoder = trm.Encoder(
            lambda: trm.EncoderLayer(
                self.embedding_dim,
                trm.MultiHeadedAttentionWithRelations(
                    self.trm_nhead,
                    self.embedding_dim,
                    self.trm_attn_dropout),
                trm.PositionwiseFeedForward(
                    self.embedding_dim,
                    self.trm_ff_dim,
                    self.trm_hidden_dropout),
                num_relation_kinds=0,
                dropout=self.trm_hidden_dropout),
            self.trm_nlayer,
            self.embedding_range,
            tie_layers=False)

    def convert_mask_trm(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def forward(self, local_embedding, global_embedding):
        local_msg = torch.relu(local_embedding)
        predict1 = self.fc(local_msg)

        batch_size, neighbor_size, emb_size = local_embedding.size()
        attention_mask = torch.ones(batch_size, neighbor_size + 1).bool().to(self.device)
        second_local = torch.cat([self.global_cls.expand(batch_size, 1, emb_size), local_embedding], dim=1)
        pos = self.pos_embeds(torch.arange(0, 3).to(self.device))
        second_local[:, 0] = second_local[:, 0] + pos[0].unsqueeze(0)
        second_local[:, 1] = second_local[:, 1] + pos[1].unsqueeze(0)
        second_local[:, 2:] = second_local[:, 2:] + pos[2].view(1, 1, -1)
        second_local = self.layer_norm(second_local)
        second_local = self.transformer_encoder(second_local, None, self.convert_mask_trm(attention_mask))
        second_local = second_local[-1][:, :2][:, 0].unsqueeze(1)
        predict2 = self.fc(torch.relu(second_local))
        predict3 = self.fc(torch.relu(global_embedding))

        predict = torch.cat([predict1, predict2, predict3], dim=1)
        weight = torch.softmax(self.temperature * predict, dim=1)
        predict = (predict * weight.detach()).sum(1).sigmoid()

        return predict

