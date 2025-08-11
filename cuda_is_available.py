# import numpy as np
# import torch
# from torch.utils.tensorboard import SummaryWriter
# from ultralytics import settings

# # # 修改设置
# # settings["tensorboard"] = True

# # # 查看当前所有设置
# # print(settings)

# print("NumPy version:", np.__version__)
# print("PyTorch version:", torch.__version__)
# print("PyTorch cuda version:", torch.version.cuda)
# print(torch.cuda.is_available())

import torch
import math
import torch.nn.functional as F

def attention(query, key, value, dropout=None):
    """
    计算缩放点积注意力
    
    Args:
        query: 查询张量，形状为 (batch_size, num_heads, seq_len, d_k)
        key: 键张量，形状为 (batch_size, num_heads, seq_len, d_k)
        value: 值张量，形状为 (batch_size, num_heads, seq_len, d_v)
        dropout: 可选的 Dropout 层

    Returns:
        输出张量和注意力权重矩阵
    """
    # 获取键向量的维度 d_k
    d_k = query.size(-1)

    # 计算 Q 和 K 的内积，并除以 sqrt(d_k) 进行缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 对得分进行 softmax 激活
    p_attn = F.softmax(scores, dim=-1)

    # 应用 Dropout（如果传入）
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 使用注意力权重对 V 进行加权求和
    output = torch.matmul(p_attn, value)

    return output, p_attn


# 示例：模拟输入 Q、K、V
batch_size = 1
num_heads = 1
seq_len = 4
d_k = 64
d_v = 64

# 随机生成 Q、K、V 张量
query = torch.randn(batch_size, num_heads, seq_len, d_k)
key = torch.randn(batch_size, num_heads, seq_len, d_k)
value = torch.randn(batch_size, num_heads, seq_len, d_v)

# 创建 Dropout 层（可选）
dropout = torch.nn.Dropout(p=0.1)

# 调用注意力函数
output, attn_weights = attention(query, key, value, dropout)

print("输出张量形状:", output.shape)
print("注意力权重形状:", attn_weights.shape)