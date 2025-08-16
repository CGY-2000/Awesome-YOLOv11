# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LEASim(nn.Module):
#     """
#     Learnable Energy Aggregation for SimAM
#     - 三种能量项: 局部一致性、通道一致性、边缘强度
#     - 自适应权重: 通道级极小 MLP → softplus → L1 归一
#     - 通道温度 gamma: 可学习标量, 控制门控锐度
#     """
#     def __init__(self, channels, reduction=4, eps=1e-6):
#         super().__init__()
#         self.c = channels
#         self.eps = eps

#         # --- Local mean: depthwise 3x3 avg (用固定均值卷积实现，groups=C)
#         kernel = torch.ones(1, 1, 3, 3) / 9.0
#         self.register_buffer("avg_kernel", kernel)

#         # Sobel kernels (固定参数，groups=C)
#         sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
#         sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
#         self.register_buffer("sobel_x", sobel_x)
#         self.register_buffer("sobel_y", sobel_y)

#         # 通道统计 -> 3个权重 (w1,w2,w3)
#         # 输入特征: per-channel mean & std (2C) 先经1x1降维 (C // reduction)，再升到 3C，再 reshape 为 (B,3,C,1,1)
#         hidden = max(channels // reduction, 4)
#         self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1, bias=True)
#         self.fc2 = nn.Conv1d(hidden, 3 * channels, kernel_size=1, bias=True)

#         # 通道温度参数 gamma_c
#         self.gamma = nn.Parameter(torch.ones(channels, 1, 1))

#     def _depthwise_conv(self, x, kernel):
#         # x: [B,C,H,W], kernel: [1,1,k,k]  → 使用 groups=C 实现深度可分离
#         B, C, H, W = x.shape
#         weight = kernel.expand(C, 1, kernel.size(2), kernel.size(3))
#         return F.conv2d(x, weight, bias=None, stride=1, padding=kernel.size(2)//2, groups=C)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         # --- 局部均值 & 局部能量
#         m_local = self._depthwise_conv(x, self.avg_kernel)                  # [B,C,H,W]
#         E_local = (x - m_local).pow(2)

#         # --- 通道均值 & 通道能量
#         mu_c = x.mean(dim=(2,3), keepdim=True)                              # [B,C,1,1]
#         E_channel = (x - mu_c).pow(2)

#         # --- Sobel 边缘能量
#         Gx = self._depthwise_conv(x, self.sobel_x)
#         Gy = self._depthwise_conv(x, self.sobel_y)
#         E_edge = Gx.pow(2) + Gy.pow(2)

#         # --- 自适应融合权重 (来自通道统计: mean & std)
#         # stats: [B,C,2]  -> 1D conv 通道维度当作“序列长度”
#         std_c = x.var(dim=(2,3), keepdim=True, unbiased=False).add(self.eps).sqrt()
#         stats = torch.cat([mu_c, std_c], dim=2).view(B, C, 2)               # [B,C,2]
#         stats = stats.transpose(1, 2)                                       # [B,2,C]
#         h = F.relu(self.fc1(stats))                                         # [B,hidden,C]
#         w = F.softplus(self.fc2(h))                                         # [B,3C,C]
#         w = w.view(B, 3, C, 1, 1)                                           # [B,3,C,1,1]
#         w = w / (w.sum(dim=1, keepdim=True) + self.eps)                     # L1 归一

#         # --- 融合能量
#         E = w[:,0]*E_local + w[:,1]*E_channel + w[:,2]*E_edge               # [B,C,H,W]

#         # --- 通道内标准化
#         mu_E = E.mean(dim=(2,3), keepdim=True)
#         std_E = E.var(dim=(2,3), keepdim=True, unbiased=False).add(self.eps).sqrt()
#         E_hat = (E - mu_E) / std_E

#         # --- 门控 & 输出 (残差)
#         gate = torch.sigmoid(-self.gamma * E_hat)                           # [C,1,1]广播
#         y = x * gate + x
#         return y


# if __name__ == '__main__':
#     # 输入张量：形状为 (B, C, H, W)
#     x = torch.randn(1, 32, 64, 64)# 例如 batch=1, 通道=32, 高=64, 宽=64
#     # 初始化 LEASim 模块
#     leasim = LEASim(channels = 32)
#     # 前向传播测试
#     output = leasim(x)
#     # 输出结果形状
#     print(leasim)
#     print("输入张量形状:", x.shape)# [B, C, H, W]  
#     print("输出张量形状:", output.shape)# [B, C, H, W] 

import torch
import torch.nn as nn
import torch.nn.functional as F

class LEASim(nn.Module):
    """
    Learnable Energy Aggregation for SimAM (fixed)
    - 使用 per-channel MLP (nn.Linear) 计算每个通道的 (w1,w2,w3)
    - 计算三类能量：局部一致性、通道一致性、边缘强度
    - 返回: y = x * gate + x
    """
    def __init__(self, channels, reduction=4, eps=1e-6):
        super().__init__()
        self.c = channels
        self.eps = eps

        # --- Local mean: depthwise 3x3 avg kernel (registered buffer)
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        self.register_buffer("avg_kernel", kernel)

        # Sobel kernels (registered buffers)
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Per-channel tiny MLP (Linear): 输入 (mean, std) -> hidden -> 3 weights
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(2, hidden, bias=True)   # applied per-channel
        self.fc2 = nn.Linear(hidden, 3, bias=True)   # outputs 3 scalars per channel

        # Channel temperature gamma_c (learnable)
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))

        # For visualization hooks (optional)
        self.last_gate = None

    def _depthwise_conv(self, x, kernel):
        # x: [B,C,H,W], kernel: [1,1,k,k] -> expand to [C,1,k,k] and groups=C
        B, C, H, W = x.shape
        # ensure kernel on same device/dtype as x
        k = kernel.to(x.device).to(x.dtype)
        weight = k.expand(C, 1, k.size(2), k.size(3))
        return F.conv2d(x, weight, bias=None, stride=1, padding=k.size(2)//2, groups=C)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 局部均值 & 局部能量
        m_local = self._depthwise_conv(x, self.avg_kernel)    # [B,C,H,W]
        E_local = (x - m_local).pow(2)

        # --- 通道均值 & 通道能量
        mu_c = x.mean(dim=(2,3))                              # [B,C]
        E_channel = (x - mu_c.view(B, C, 1, 1)).pow(2)        # [B,C,H,W]

        # --- Sobel 边缘能量
        Gx = self._depthwise_conv(x, self.sobel_x)
        Gy = self._depthwise_conv(x, self.sobel_y)
        E_edge = Gx.pow(2) + Gy.pow(2)                        # [B,C,H,W]

        # --- 自适应融合权重 (来自通道统计: mean & std)，用 per-channel MLP
        std_c = x.var(dim=(2,3), keepdim=False, unbiased=False).add(self.eps).sqrt()  # [B,C]
        stats = torch.stack([mu_c, std_c], dim=2)             # [B,C,2]

        # apply per-channel MLP: treat last dim as features
        h = F.relu(self.fc1(stats))                           # [B,C,hidden]
        w = F.softplus(self.fc2(h))                           # [B,C,3]

        # reshape -> [B,3,C,1,1] for broadcasting with E_*
        w = w.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)    # [B,3,C,1,1]
        w = w / (w.sum(dim=1, keepdim=True) + self.eps)       # L1 归一化 across the 3 weights

        # --- 融合能量
        E = w[:,0] * E_local + w[:,1] * E_channel + w[:,2] * E_edge   # [B,C,H,W]

        # --- 通道内标准化
        mu_E = E.mean(dim=(2,3), keepdim=True)                # [B,C,1,1]
        std_E = E.var(dim=(2,3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        E_hat = (E - mu_E) / std_E

        # --- 门控 & 输出 (残差)
        gate = torch.sigmoid(-self.gamma * E_hat)             # [B,C,H,W], gamma broadcasts
        # save for visualization hooks (copy to CPU to avoid GPU memory hold)
        try:
            self.last_gate = gate.detach().cpu()
        except Exception:
            self.last_gate = None

        y = x * gate + x
        return y


if __name__ == '__main__':
    # 测试
    x = torch.randn(1, 32, 64, 64)  # [B, C, H, W]
    leasim = LEASim(channels=32)
    output = leasim(x)
    print(leasim)
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
    # 验证 last_gate 形状
    if leasim.last_gate is not None:
        print("last_gate shape (cpu):", leasim.last_gate.shape)  # [B,C,H,W]
