import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
# from mamba_ssm import Mamba  #
import torch
import torch.nn as nn
import pywt  #
import math
# from mamba_ssm import Mamba  #  mamba-ssm



# new 0

class MultiScaleDWFormer(nn.Module):
    def __init__(self, input_size=50, embed_dim=128, num_heads=4, num_classes=2):
        super(MultiScaleDWFormer, self).__init__()

        # Step 1: Multi-scale CNN
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 64, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(64 * 3)

        # Step 2: Channel-wise attention (dynamic weighting)
        self.fc_attention = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * 3),
            nn.Sigmoid()
        )

        # Step 3: Transformer encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, input_size, embed_dim))
        self.embedding = nn.Linear(64 * 3, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len]
        x = x.unsqueeze(1)  # [B, 1, seq_len]

        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)

        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 64*3, seq_len]
        x_cat = self.bn(x_cat)

        # Permute for attention: [B, seq_len, C]
        x_cat = x_cat.permute(0, 2, 1)  # [B, seq_len, 192]
        attn_weights = self.fc_attention(x_cat.mean(dim=1))  # [B, 192]
        attn_weights = attn_weights.unsqueeze(1)  # [B, 1, 192]
        x_weighted = x_cat * attn_weights  # [B, seq_len, 192]

        # Embedding + Positional Encoding
        x_emb = self.embedding(x_weighted) + self.pos_embedding[:, :x_weighted.size(1), :]  # [B, seq_len, embed_dim]

        x_trans = self.transformer(x_emb)  # [B, seq_len, embed_dim]
        x_pool = x_trans.mean(dim=1)  # [B, embed_dim]

        out = self.classifier(x_pool)
        return out





# =========================
# Ablation 1: Remove Step 1 (Multi-scale CNN)
# =========================
class Ablation_NoStep1(nn.Module):
    def __init__(self, input_size=50, embed_dim=128, num_heads=4, num_classes=2):
        super(Ablation_NoStep1, self).__init__()
        # Step 1 removed
        # Step 2: attention now input_size -> 64 -> input_size
        self.fc_attention = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Step 3
        self.pos_embedding = nn.Parameter(torch.randn(1, input_size, embed_dim))
        self.embedding = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_cat = x.unsqueeze(-1)  # [B, seq_len, 1]
        attn_weights = self.fc_attention(x_cat.mean(dim=1)).unsqueeze(1)  # [B, 1, 1]
        x_weighted = x_cat * attn_weights
        x_emb = self.embedding(x_weighted) + self.pos_embedding[:, :x_weighted.size(1), :]
        x_trans = self.transformer(x_emb)
        x_pool = x_trans.mean(dim=1)
        out = self.classifier(x_pool)
        return out



# =========================
# Ablation 2: Remove Step 2 (Dynamic Attention)
# =========================
class Ablation_NoStep2(nn.Module):
    def __init__(self, input_size=50, embed_dim=128, num_heads=4, num_classes=2):
        super(Ablation_NoStep2, self).__init__()
        # Step 1: Multi-scale CNN
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(64 * 3)

        # Step 2 removed

        # Step 3: Transformer encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, input_size, embed_dim))
        self.embedding = nn.Linear(64 * 3, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_cat = self.bn(x_cat)
        x_weighted = x_cat.permute(0, 2, 1)  # skip attention
        x_emb = self.embedding(x_weighted) + self.pos_embedding[:, :x_weighted.size(1), :]
        x_trans = self.transformer(x_emb)
        x_pool = x_trans.mean(dim=1)
        out = self.classifier(x_pool)
        return out


# =========================
# Ablation 3: Remove Step 3 (Transformer encoder)
# =========================
class Ablation_NoStep3(nn.Module):
    def __init__(self, input_size=50, embed_dim=128, num_heads=4, num_classes=2):
        super(Ablation_NoStep3, self).__init__()
        # Step 1: Multi-scale CNN
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(64 * 3)

        # Step 2: Dynamic Attention
        self.fc_attention = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * 3),
            nn.Sigmoid()
        )

        # Step 3 removed
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_cat = self.bn(x_cat)
        x_cat = x_cat.permute(0, 2, 1)
        attn_weights = self.fc_attention(x_cat.mean(dim=1)).unsqueeze(1)
        x_weighted = x_cat * attn_weights
        x_pool = x_weighted.mean(dim=1)  # 平均池化代替Transformer
        out = self.classifier(x_pool)
        return out





# new 1
class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim=50, embed_dim=128, num_heads=8, num_layers=4, num_classes=2):
        super().__init__()

        # 1. 多尺度特征提取
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.MaxPool1d(2)
        )

        # 2. 位置编码 + Transformer
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 3. 注意力池化（替代全局平均）
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # 输入形状: [B, 50]
        x = x.unsqueeze(1)  # [B, 1, 50]

        # 1. 卷积特征提取
        conv_out = self.conv_block(x)  # [B, embed_dim, 25]
        conv_out = conv_out.permute(0, 2, 1)  # [B, 25, embed_dim]

        # 2. Transformer处理
        x = self.pos_encoder(conv_out)
        x = self.transformer(x)  # [B, 25, embed_dim]

        # 3. 注意力池化
        attn_weights = self.attention_pool(x)  # [B, 25, 1]
        x = torch.sum(x * attn_weights, dim=1)  # [B, embed_dim]

        # 4. 分类
        return self.classifier(x)



# new  2
class TimeFreqNet(nn.Module):
    def __init__(self, input_dim=50, num_classes=2):
        super().__init__()

        # 时域分支
        self.time_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # 频域分支（模拟FFT）
        self.freq_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        # 融合模块
        self.fusion = nn.Linear(256, 128)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # 时域处理
        x_time = x.unsqueeze(1)  # [B, 1, 50]
        x_time = self.time_net(x_time)  # [B, 128, 25]
        x_time = x_time.mean(dim=2)  # [B, 128]

        # 频域处理
        x_freq = self.freq_net(x)  # [B, 128]

        # 特征融合
        x = torch.cat([x_time, x_freq], dim=1)  # [B, 256]
        x = self.fusion(x).unsqueeze(1)  # [B, 1, 128]

        # Transformer处理
        x = self.transformer(x)  # [B, 1, 128]

        return self.classifier(x.squeeze(1))


# new  3
class CapsuleLayer(nn.Module):
    def __init__(self, in_caps, out_caps, in_dim, out_dim, device):
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.device = device

        # 初始化权重矩阵 [in_caps, out_caps, in_dim, out_dim]
        self.W = nn.Parameter(torch.randn(in_caps, out_caps, in_dim, out_dim))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        # x: [B, in_caps, in_dim]
        batch_size = x.size(0)

        # 扩展维度用于矩阵乘法 [B, in_caps, out_caps, in_dim, out_dim]
        x = x.unsqueeze(2).unsqueeze(3)  # [B, in_caps, 1, 1, in_dim]
        W = self.W.unsqueeze(0)  # [1, in_caps, out_caps, in_dim, out_dim]

        # 计算预测向量 u_hat [B, in_caps, out_caps, out_dim]
        u_hat = torch.matmul(x, W).squeeze(3)  # [B, in_caps, out_caps, out_dim]

        # 动态路由算法
        b = torch.zeros(batch_size, self.in_caps, self.out_caps, 1).to(self.device)

        for i in range(3):  # 路由迭代3次
            # 计算耦合系数c [B, in_caps, out_caps, 1]
            c = F.softmax(b, dim=2)

            # 计算加权和 [B, out_caps, out_dim]
            s = (c * u_hat).sum(dim=1)

            # 非线性压缩
            v = self.squash(s)

            # 更新bij
            if i < 2:  # 最后一次迭代不需要更新
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)

        return v  # [B, out_caps, out_dim]


class CapsTransformer(nn.Module):
    def __init__(self, input_dim=50, num_classes=2, device='cuda'):
        super().__init__()
        self.device = device

        # 1. 特征嵌入
        self.embedding = nn.Linear(input_dim, 128)

        # 2. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256,
            batch_first=True, device=device
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. 初级胶囊层 (32个胶囊，每个8维)
        self.primary_caps = nn.Sequential(
            nn.Linear(128, 32 * 8),
            nn.LayerNorm(32 * 8)
        )

        # 4. 数字胶囊层 (num_classes个胶囊，每个16维)
        self.digit_caps = CapsuleLayer(
            in_caps=32,
            out_caps=num_classes,
            in_dim=8,
            out_dim=16,
            device=device
        )

    def forward(self, x):
        # 输入形状: [B, input_dim]
        x = x.unsqueeze(1)  # [B, 1, input_dim]

        # 1. 特征嵌入
        x = self.embedding(x)  # [B, 1, 128]

        # 2. Transformer处理
        x = self.transformer(x)  # [B, 1, 128]

        # 3. 初级胶囊
        x = self.primary_caps(x)  # [B, 1, 256]
        x = x.view(x.size(0), 32, 8)  # [B, 32, 8]

        # 4. 数字胶囊
        x = self.digit_caps(x)  # [B, num_classes, 16]

        # 5. 用向量长度表示类别概率
        return torch.norm(x, dim=-1)  # [B, num_classes]




# new 4
class MemoryModule(nn.Module):
    def __init__(self, mem_dim, mem_size):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))

    def forward(self, x):
        # x: [B, D]
        sim = torch.matmul(x, self.memory.T)  # [B, M]
        attn = F.softmax(sim, dim=1)
        return torch.matmul(attn, self.memory)  # [B, D]


class MANet(nn.Module):
    def __init__(self, input_dim=50, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.memory = MemoryModule(mem_dim=128, mem_size=64)

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),  # 拼接原始特征和记忆特征
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)  # [B, 128]
        mem_feat = self.memory(feat)  # [B, 128]
        combined = torch.cat([feat, mem_feat], dim=1)  # [B, 256]
        return self.decoder(combined)




# new 5  error
class ConvNeXt1D(nn.Module):
    def __init__(self, input_dim=50, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.LayerNorm(64)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Conv1d(64, 256, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(256, 64, kernel_size=1)
            ) for _ in range(3)]
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 50]
        x = self.stem(x).permute(0, 2, 1)  # 通道最后
        x = self.blocks(x)
        return self.head(x.mean(dim=1))



# best 1 model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class TriFormerNet(nn.Module):
    def __init__(self, seq_len=50, embed_dim=128, num_heads=4, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # 多尺度卷积
        self.conv3 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.norm_conv = nn.LayerNorm(96)

        # ConvNeXtBlock 替代传统卷积块
        self.convnext_block = ConvNeXtBlock(96)

        # 投影 + Transformer
        self.embedding = nn.Linear(96, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 输入 x: [B, seq_len]
        x = x.unsqueeze(1)  # [B, 1, seq_len]
        x3 = F.gelu(self.conv3(x))
        x5 = F.gelu(self.conv5(x))
        x7 = F.gelu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)  # [B, 96, seq_len]
        x = x.permute(0, 2, 1)  # [B, seq_len, 96]
        x = self.norm_conv(x)
        x = self.convnext_block(x)

        x = self.embedding(x)  # [B, seq_len, embed_dim]
        x = self.pos_encoding(x)
        x = self.transformer(x)

        x = x.mean(dim=1)  # 全局池化
        out = self.classifier(x)
        return out


# best 2 model
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, L, D]
        weights = torch.softmax(self.attn(x), dim=1)  # [B, L, 1]
        return (x * weights).sum(dim=1)  # [B, D]

class TriFormerV2(nn.Module):
    def __init__(self, seq_len=50, embed_dim=128, num_heads=4, num_layers=1, num_classes=2, dropout=0.1):
        super().__init__()
        # 多尺度卷积
        self.conv3 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.norm_conv = nn.LayerNorm(96)

        # ConvNeXtBlock
        self.convnext_block = ConvNeXtBlock(96)

        # 嵌入 + 位置编码
        self.embedding = nn.Linear(96, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=seq_len)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 注意力池化
        self.attn_pool = AttentionPooling(embed_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x3 = F.gelu(self.conv3(x))
        x5 = F.gelu(self.conv5(x))
        x7 = F.gelu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = x.permute(0, 2, 1)
        x = self.norm_conv(x)
        x = self.convnext_block(x)

        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        x = self.attn_pool(x)
        out = self.classifier(x)
        return out



#improved：HybridMemoryFormer
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w.expand_as(x)


class HybridMemoryFormer(nn.Module):
    def __init__(self, input_dim=50, embed_dim=128, num_heads=4, num_layers=2, num_classes=2):
        super().__init__()

        # 多尺度时域卷积块 + SE 注意力
        self.conv_block = nn.Sequential(
            ConvBlock(1, 32),
            SEBlock(32),
            nn.MaxPool1d(2),
            ConvBlock(32, embed_dim),
            SEBlock(embed_dim)
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 记忆增强模块
        self.memory = MemoryModule(mem_dim=embed_dim, mem_size=64)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        feat = self.conv_block(x)  # [B, C, L/2]
        feat = feat.transpose(1, 2)  # [B, L/2, C]
        x_trans = self.transformer(feat)  # [B, L/2, C]
        x_pool = x_trans.mean(dim=1)  # 全局池化
        mem_feat = self.memory(x_pool)
        combined = torch.cat([x_pool, mem_feat], dim=1)
        return self.classifier(combined)



# Improved Second Edition
import torch
import torch.nn as nn
import torch.nn.functional as F

# MultiScaleConv
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.act = nn.GELU()

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x_cat = torch.cat([x3, x5, x7], dim=1)
        return self.act(self.norm(x_cat))

# ConvNeXt Block
class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

# GatedMemoryModule
class GatedMemoryModule(nn.Module):
    def __init__(self, mem_dim=128, mem_size=64):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))
        self.gate = nn.Sequential(
            nn.Linear(mem_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sim = torch.matmul(x, self.memory.T)
        attn = F.softmax(sim, dim=1)
        read = torch.matmul(attn, self.memory)
        g = self.gate(x)
        return g * read + (1 - g) * x  # 门控融合机制

# AttentionPooling
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

# Main Model
class HybridMemoryFormerV2(nn.Module):
    def __init__(self, input_len=50, in_channels=1, embed_dim=128, num_heads=4,
                 num_layers=2, mem_dim=128, mem_size=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv = MultiScaleConv(in_channels, 32)
        self.convnext = ConvNeXt1DBlock(96)

        self.linear_proj = nn.Linear(96, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.memory = GatedMemoryModule(mem_dim=embed_dim, mem_size=mem_size)
        self.pool = AttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)    # [B, 96, L]
        x = x.permute(0, 2, 1)  # [B, L, 96]
        x = self.convnext(x)    # [B, L, 96]

        x = self.linear_proj(x)     # [B, L, embed_dim]
        x = self.transformer(x)     # [B, L, embed_dim]
        mem = self.memory(x.mean(dim=1))  # [B, embed_dim]
        x_pool = self.pool(x)            # [B, embed_dim]

        x_fused = x_pool + mem  # 简单融合（也可考虑 concat + fc）
        return self.classifier(x_fused)
