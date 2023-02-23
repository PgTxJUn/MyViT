import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Patch Embedding 模块

    img_size: 输入图片的尺寸。

    patch_size: 每一个patch的尺寸。

    in_ch：输入的通道数。

    embed_dim: 维度数。

    n_patches: 在图像中patch的数量。

    proj: 卷积层。
    """

    def __init__(self, img_size, patch_size, in_ch=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """
    注意力机制模块

    dim: 每一个token输入和输出的特征维度

    n_heads: 多头注意力机制的头数量

    qkv_bias: Q、K、V矩阵中是否有偏差

    attn_p: Q、K、V矩阵dropout的概率

    proj_p: 输出dropout的概率

    scale: 点乘所需要的比例因子

    qkv: QKV线性层

    proj: 所有注意力头后的线性层

    attn_drop, proj_drop: dropout层
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def forward(self, x):
        pass
