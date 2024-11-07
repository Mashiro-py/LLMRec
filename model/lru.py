import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .utils import Transformer,CrossModule
from .modules import *


class LRURec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = LRU_LLM_Embedding_mix(self.args)
        self.model = LRUModel(self.args)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, x):
        x, mask = self.embedding(x)
        scores = self.model(x, self.embedding.token.weight, mask)
        return scores


class FeatureAlignment(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 特征变换层
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 可学习的权重
        self.alpha = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, x):
        # 特征对齐
        aligned = self.alignment(x)
        # 特征归一化
        return F.normalize(aligned, p=2, dim=-1)


class HierarchicalFeatures(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 低层特征提取
        self.low_level = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        # 高层特征提取
        self.high_level = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, id_emb, text_emb):
        # 提取低层特征
        low_features = self.low_level(id_emb)

        # 提取高层特征
        high_features = self.high_level(text_emb)

        # 特征拼接
        combined = torch.cat([low_features, high_features], dim=-1)

        # 计算融合门控权重
        gate = self.fusion_gate(combined)

        # 加权融合
        fused = gate * low_features + (1 - gate) * high_features

        return F.normalize(fused, p=2, dim=-1)


class LRU_LLM_Embedding_mix(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vocab_size = args.num_items + 2
        embed_size = args.bert_hidden_units

        # ID Embedding
        self.token = nn.Embedding(vocab_size, embed_size)

        # 文本特征
        self.text_embeddings = nn.Embedding(vocab_size, args.pretrain_emb_dim, padding_idx=0)

        # 特征对齐层
        self.feature_alignment = FeatureAlignment(
            input_dim=args.pretrain_emb_dim,
            hidden_dim=embed_size
        )

        # 分层特征提取
        self.hierarchical_features = HierarchicalFeatures(embed_size)

        # 特征转换层
        self.transform = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.LayerNorm(embed_size)
        )

        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(args.bert_dropout)

        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.FloatTensor([0.1]))

        if args.is_use_mm:
            print("---------- Loading Multimodal Features -----------")
            self.init_mm_features()

        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for name, p in self.named_parameters():
            if 'layer_norm' in name:
                continue
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'fusion_weight' in name:
                nn.init.constant_(p, 0.1)
            elif len(p.shape) >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def init_mm_features(self):
        """初始化多模态特征"""
        text_features = torch.load(self.args.text_embedding_path)
        self.text_embeddings.weight.data[1:-1, :] = text_features

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)

        # 1. 获取ID embedding
        item_embeddings = self.token(x)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)

        if self.args.is_use_text:
            try:
                # 2. 获取文本特征
                text_emb = self.text_embeddings(x)

                # 3. 特征对齐和归一化
                aligned_text = self.feature_alignment(text_emb)

                # 4. 分层特征提取和融合
                fused_features = self.hierarchical_features(item_embeddings, aligned_text)

                # 5. 特征转换
                fused_features = self.transform(fused_features)

                # 6. 动态加权组合
                fusion_gate = torch.sigmoid(self.fusion_weight)
                item_embeddings = fusion_gate * item_embeddings + (1 - fusion_gate) * fused_features

                # 7. 最终归一化
                item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)

            except Exception as e:
                print(f"Error in fusion process: {e}")
                return item_embeddings, mask

        # 8. Dropout和Layer Norm
        output = self.layer_norm(self.dropout(item_embeddings))

        return output, mask



class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, item_emb, text_emb):
        # [B, L, D] -> [L, B, D]
        item_emb = item_emb.transpose(0, 1)
        text_emb = text_emb.transpose(0, 1)

        # 交叉注意力
        output, _ = self.cross_attention(item_emb, text_emb, text_emb)
        # [L, B, D] -> [B, L, D]
        return output.transpose(0, 1)



class LRU_Cross_LLM_Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vocab_size = args.num_items + 2
        embed_size = args.bert_hidden_units

        # Embeddings
        self.token = nn.Embedding(vocab_size, embed_size)
        self.text_embeddings = nn.Embedding(vocab_size, args.pretrain_emb_dim, padding_idx=0)

        # 特征映射层
        self.fc_text = nn.Linear(args.pretrain_emb_dim, embed_size)

        # 交叉注意力融合层
        self.cross_fusion = CrossAttentionFusion(embed_size)

        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

        if args.is_use_mm:
            print("---------- Loading Multimodal Features -----------")
            self.init_mm_features()

    def init_mm_features(self):
        """初始化多模态特征"""
        text_features = torch.load(self.args.text_embedding_path)
        self.text_embeddings.weight.data[1:-1, :] = text_features

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        item_embeddings = self.token(x)

        if self.args.is_use_text:
            # 获取文本特征
            text_emb = self.text_embeddings(x)
            text_emb = self.fc_text(text_emb)
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            # 交叉注意力融合
            item_embeddings = self.cross_fusion(item_embeddings, text_emb)

        x = self.layer_norm(self.embed_dropout(item_embeddings))
        return x, mask


class LRUEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units

        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        x = self.token(x)
        return self.layer_norm(self.embed_dropout(x)), mask



class LRU_Attention_LLM_Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))
        vocab_size = args.num_items + 2
        embed_size = args.bert_hidden_units

        # ID  Embedding
        self.token = nn.Embedding(vocab_size, embed_size)
        # 文本特征
        self.text_embeddings = nn.Embedding(vocab_size, args.pretrain_emb_dim, padding_idx=0)
        # 特征映射层
        self.fc_text = nn.Linear(args.pretrain_emb_dim, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)
        self.cross_module = CrossModule()

        if args.is_use_mm:
            print("---------- Loading Multimodal Features -----------")
            self.init_mm_features()

    def init_mm_features(self):
        """初始化多模态特征"""
        text_features = torch.load(self.args.text_embedding_path)
        self.text_embeddings.weight.data[1:-1, :] = text_features
    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        item_embeddings = self.token(x)  # [B, L, D]

        if self.args.is_use_text:
            # 获取文本特征
            text_emb = self.text_embeddings(x)  # [B, L, D]
            text_emb = self.fc_text(text_emb)  # [B, L, D]
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            # 计算注意力分数
            attention_scores = torch.matmul(item_embeddings, text_emb.transpose(-2, -1))  # [B, L, L]
            attention_scores = attention_scores / math.sqrt(item_embeddings.size(-1))
            attention_weights = F.softmax(attention_scores, dim=-1)  # [B, L, L]

            # 加权融合
            fused_embeddings = torch.matmul(attention_weights, text_emb)
            # 使用alpha进行加权组合
            item_embeddings = (1 - self.alpha) * item_embeddings + self.alpha * fused_embeddings

        x = self.layer_norm(self.embed_dropout(item_embeddings))
        return x, mask



class LRU_LLM_Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vocab_size = args.num_items + 2
        embed_size = args.bert_hidden_units

        # ID  Embedding
        self.token = nn.Embedding(vocab_size, embed_size)
        # 文本特征
        self.text_embeddings = nn.Embedding(vocab_size, args.pretrain_emb_dim, padding_idx=0)
        # 特征映射层
        self.fc_text = nn.Linear(args.pretrain_emb_dim, embed_size)
        # 多模态融合层
        self.trans = Transformer(model_dimension=embed_size,
                                 number_of_heads=4,
                                 number_of_layers=1,
                                 dropout_probability=0.1)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)
        self.cross_module = CrossModule()

        if args.is_use_mm:
            print("---------- Loading Multimodal Features -----------")
            self.init_mm_features()

    def init_mm_features(self):
        """初始化多模态特征"""
        text_features = torch.load(self.args.text_embedding_path)
        self.text_embeddings.weight.data[1:-1, :] = text_features
    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        item_embeddings = self.token(x)
        if self.args.is_use_text:
            # 获取文本特征
            text_emb = self.text_embeddings(x)

            # 特征映射
            text_emb = self.fc_text(text_emb)

            # 特征归一化
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            # 特征融合
            item_embeddings = item_embeddings + text_emb

        x = self.layer_norm(self.embed_dropout(item_embeddings))
        return x, mask


class LRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        layers = args.bert_num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(layers)])
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 2))

    def forward(self, x, embedding_weight, mask):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64)

        scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
        return scores


class LRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = args.bert_hidden_units
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=args.bert_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x
    

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)