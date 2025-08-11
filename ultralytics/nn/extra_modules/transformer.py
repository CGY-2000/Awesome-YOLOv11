import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .prepbn import RepBN, LinearNorm
from .attention import *
from .ast import AdaptiveSparseSA
from .filc import FMFFN
from .semnet import SEFN
from .mona import Mona
from .transMamba import SpectralEnhancedFFN
from .EVSSM import EDFFN
from ..modules.transformer import TransformerEncoderLayer, AIFI
from ..modules.block import C2PSA, PSABlock, ABlock, A2C2f, C3k

__all__ = ['AIFI_RepBN', 'C2BRA', 'C2CGA', 'C2DA', 'C2DPB', 'C2Pola', 'C2TSSA', 'C2ASSA', 'C2PSA_DYT', 'C2TSSA_DYT', 'C2Pola_DYT',
           'C2PSA_FMFFN', 'C2PSA_CGLU', 'C2PSA_SEFN', 'C2PSA_Mona', 'C2TSSA_DYT_Mona', 'C2TSSA_DYT_Mona_SEFN', 'C2PSA_SEFFN',
           'C2TSSA_DYT_Mona_SEFFN', 'C2PSA_EDFFN', 'C2TSSA_DYT_Mona_EDFFN']

ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)

class TransformerEncoderLayer_RepBN(TransformerEncoderLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=..., normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        
        self.norm1 = linearnorm(c1)
        self.norm2 = linearnorm(c1)

class AIFI_RepBN(TransformerEncoderLayer_RepBN):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class BRABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = BiLevelRoutingAttention_nchw(dim=c)

class C2BRA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(BRABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class CGABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = LocalWindowAttention(dim=c)

class C2CGA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(CGABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class DABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = DAttention(c, q_size=[20, 20])

class C2DA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(DABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## CrossFormer start ########################################

class DynamicPosBias(nn.Module):
    r"""DPB module
    
    Use a MLP to predict position bias used in attention.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class DPB_Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            
            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2Ww-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases, persistent=False)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # @ stands for matrix multiplication
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DPBlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = DPB_Attention(c, group_size=[20, 20], num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2DPB(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(DPBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
        
######################################## CrossFormer end ########################################

######################################## ICLR2025 PolaFormer start ########################################

class PolaLinearAttention(nn.Module):
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.h = hw[0]
        self.w = hw[1]
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        
        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, (self.w * self.h) // (sr_ratio * sr_ratio), dim)))

    def forward(self, x):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        
        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)
        
        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous() 
        
        q_pos = kernel_function(q) ** power 
        q_neg = kernel_function(-q) ** power 
        k_pos = kernel_function(k) ** power 
        k_neg = kernel_function(-k) ** power 

        q_sim = torch.cat([q_pos, q_neg],dim=-1)
        q_opp = torch.cat([q_neg, q_pos],dim=-1)
        k = torch.cat([k_pos, k_neg],dim=-1)

        v1,v2 = torch.chunk(v,2,dim=-1)
        
        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp],dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        
        v = v.reshape(B * self.num_heads, self.h, self.w, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Polalock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = PolaLinearAttention(c, hw=[20, 20], num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2Pola(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(Polalock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

######################################## ICLR2025 PolaFormer end ########################################

######################################## ICLR2025 Token Statistics Transformer start ########################################

class TSSAlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2TSSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(TSSAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        BS, C, H, W = b.size()
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

######################################## ICLR2025 Token Statistics Transformer end ########################################

######################################## CVPR2024 Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration start ########################################

class ASSAlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = AdaptiveSparseSA(c, num_heads=num_heads, sparseAtt=True)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2ASSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(ASSAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

######################################## ICLR2025 Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration end ########################################

######################################## CVPR2025 DynamicTanh start ########################################

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class PSABlock_DYT(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
    
    def forward(self, x):
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

class C2PSA_DYT(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class TSSAlock_DYT(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

class C2TSSA_DYT(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class Polalock_DYT(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = PolaLinearAttention(c, hw=[20, 20], num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

class C2Pola_DYT(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(Polalock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## CVPR2025 DynamicTanh end ########################################
        
######################################## ICLR2024 FTIC start ########################################

class PSABlock_FMFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = FMFFN(c, c * 2, c)

class C2PSA_FMFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_FMFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## ICLR2024 FTIC end ########################################
        
######################################## CVPR2024 Transnext start ########################################

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x

class PSABlock_CGLU(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = ConvolutionalGLU(c, c * 2, c)

class C2PSA_CGLU(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_CGLU(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## CVPR2024 Transnext end ########################################

######################################## WACV2025-SEMNet start ########################################
        
class PSABlock_SEFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SEFN(c, ffn_expansion_factor=2, bias=False)
    
    def forward(self, x):
        x_spatial = x
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x, x_spatial) if self.add else self.ffn(x, x_spatial)
        return x

class C2PSA_SEFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_SEFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## WACV2025-SEMNet end ########################################
        
######################################## CVPR2025-Mona start ########################################
        
class PSABlock_Mona(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
    
    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = self.mona1(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        x = self.mona2(x)
        return x

class C2PSA_Mona(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_Mona(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class TSSAlock_DYT_Mona(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

class C2TSSA_DYT_Mona(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class TSSAlock_DYT_Mona_SEFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SEFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x_spatial = x
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x), x_spatial) if self.add else self.ffn(self.dyt2(x), x_spatial)
        x = self.mona2(x)
        return x

class C2TSSA_DYT_Mona_SEFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_SEFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## CVPR2025-Mona end ########################################
        
######################################## TransMamba start ########################################

class PSABlock_SEFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SpectralEnhancedFFN(c, 2, False)

class C2PSA_SEFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_SEFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class TSSAlock_DYT_Mona_SEFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SpectralEnhancedFFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

class C2TSSA_DYT_Mona_SEFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_SEFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## TransMamba end ########################################
        
######################################## CVPR2025 EVSSM start ########################################
        
class PSABlock_EDFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(c, 2, False)

class C2PSA_EDFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

class TSSAlock_DYT_Mona_EDFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

class C2TSSA_DYT_Mona_EDFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

######################################## CVPR2025 EVSSM end ########################################