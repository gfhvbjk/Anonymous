import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
# MLP层
import torch
import torch.nn as nn
from torch import Tensor
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 定义 QKV 的线性层
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # 输出投影层
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape

        # 计算 Q、K、V
        qkv = self.qkv(x)  # [B, N, 3 * embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)  # 每个形状为 [B, num_heads, N, head_dim]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)

        # 计算注意力输出
        out = (attn @ v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, embed_dim]

        # 输出投影
        out = self.proj(out)
        return out
# MLP模块
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

# Transformer编码器层
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout_rate: float, attention_dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=attention_dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        # 自注意力模块
        x_res = x
        x = self.ln_1(x)
        x = self.self_attention(x)
        x = x_res + self.dropout1(x)
        # MLP模块
        x_res = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x_res + self.dropout2(x)
        return x

# Transformer编码器
class Encoder(nn.Module):
    def __init__(self, seq_length: int, hidden_dim: int, num_layers: int, num_heads: int, mlp_dim: int, dropout_rate: float, attention_dropout_rate: float):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)
        # 使用 ModuleDict 并命名每个层，与 torchvision 模型匹配
        layers = []
        for i in range(num_layers):
            layers.append((f'encoder_layer_{i}', EncoderLayer(hidden_dim, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)))
        self.layers = nn.ModuleDict(layers)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embedding
        x = self.dropout(x)
        for layer in self.layers.values():
            x = layer(x)
        x = self.ln(x)
        return x

# ViT模型定义
class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 hidden_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0):
        super().__init__()
        self.conv_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        seq_length = num_patches + 1  # 加上类别标记

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = Encoder(seq_length, hidden_dim, num_layers, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)
        # 分类头使用 ModuleDict，并命名为 'head'，与 torchvision 模型匹配
        self.heads = nn.ModuleDict({'head': nn.Linear(hidden_dim, num_classes)})

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.conv_proj.weight, std=0.02)
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.encoder.pos_embedding, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)  # [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]
        B = x.shape[0]
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)  # [B, N+1, hidden_dim]
        x = self.encoder(x)
        x = self.heads['head'](x[:, 0])
        return x


# 实例化模型
model = VisionTransformer()

# 加载 torchvision 的预训练权重
import torchvision

# 获取 torchvision 预训练模型的状态字典
pretrained_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
pretrained_dict = pretrained_model.state_dict()

# 加载预训练权重到自定义模型
model.load_state_dict(pretrained_dict)
print("Keys in model.encoder.layers:", model.encoder.layers.keys())

# 遍历模型的所有 Transformer Encoder 层，替换其中的 MultiheadAttention
for name, encoder_layer in model.encoder.layers.items():
    # 获取 MultiheadAttention 层
    mha_layer = encoder_layer.self_attention

    # 创建自定义的线性注意力层
    linear_attn = LinearAttention(embed_dim=768, num_heads=12)

    # 复制权重

    # 复制 qkv 的权重和偏置
    # 从原始 mha_layer 中提取 in_proj_weight 和 in_proj_bias
    in_proj_weight = mha_layer.in_proj_weight  # [3 * embed_dim, embed_dim]
    in_proj_bias = mha_layer.in_proj_bias  # [3 * embed_dim]

    # 将 in_proj_weight 和 in_proj_bias 复制到 linear_attn 的 qkv 层
    linear_attn.qkv.weight.data.copy_(in_proj_weight)
    linear_attn.qkv.bias.data.copy_(in_proj_bias)

    # 复制 out_proj 的权重和偏置
    linear_attn.proj.weight.data.copy_(mha_layer.out_proj.weight)
    linear_attn.proj.bias.data.copy_(mha_layer.out_proj.bias)

    # 替换原始的 MultiheadAttention 层为自定义的线性注意力层
    encoder_layer.self_attention = linear_attn

# print("所有的 MultiheadAttention 层已成功替换为自定义的线性注意力层。")
#
#
#
# print("预训练权重加载成功！")
# print(model)
# # 测试模型的前向传播
# dummy_input = torch.randn(1, 3, 224, 224)
# output = model(dummy_input)
# print("模型输出形状：", output.shape)
