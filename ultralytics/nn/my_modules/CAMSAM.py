import torch
import torch.nn as nn
import torch.nn.functional as F

class CAMSAM(nn.Module):
    """
    Context-Aware Multi-Scale Attention Module (CAMS-AM)
    
    This module enhances SimAM by considering multi-scale local contexts 
    instead of a single global context. It computes energy functions based on
    statistics from different local neighborhood sizes and aggregates them 
    to produce a more robust attention map.
    """
    # ----- FIX: Removed the stray hyphen before 'lambda_' -----
    def __init__(self, channels=None, kernel_sizes=[3, 7, 11], lambda_=1e-4):
        """
        Args:
            channels (int): Number of input channels. Not strictly used in the 
                            parameter-free calculation but good for consistency with other modules.
            kernel_sizes (list of int): List of odd integers for kernel sizes of local windows.
            lambda_ (float): Regularization coefficient.
        """
        super(CAMSAM, self).__init__()
        
        if not all(k % 2 == 1 for k in kernel_sizes):
            raise ValueError("All kernel_sizes must be odd integers.")
            
        self.kernel_sizes = kernel_sizes
        self.lambda_ = lambda_
        
        # Use nn.AvgPool2d to efficiently calculate local means
        # 'padding' is set to (k // 2) to keep the spatial dimensions the same.
        self.avg_pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size, stride=1, padding=(kernel_size // 2))
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Enhanced feature map with the same shape as input.
        """
        b, c, h, w = x.shape
        
        # Store energy maps for each scale
        energy_maps = []

        for i, kernel_size in enumerate(self.kernel_sizes):
            # Calculate local mean (μ) using average pooling
            # This is equivalent to a sliding window mean.
            local_mean = self.avg_pools[i](x) # (B, C, H, W)
            
            # Calculate local variance (σ^2)
            # Var(X) = E[X^2] - (E[X])^2
            local_var = self.avg_pools[i](x.pow(2)) - local_mean.pow(2) # (B, C, H, W)

            # Calculate the energy function for each pixel (neuron)
            # The original SimAM energy function for a target neuron 't' is:
            # e_t = (1/(M-1)) * Σ( (t - μ_t)^2 + (t - x_i)^2 ) + λ*(t-μ_t)^2
            # A simplified and efficient form is used:
            # e_t = (t - μ)^2 / (4 * (σ^2 + λ)) + 0.5
            # Here, we use this simplified form within each local context.

            numerator = (x - local_mean).pow(2)
            denominator = 4 * (local_var + self.lambda_)
            
            energy = numerator / denominator + 0.5
            energy_maps.append(energy)

        # Aggregate energy maps from all scales.
        # Simple averaging is a robust choice.
        # We sum them up and then divide by the number of scales.
        fused_energy = torch.stack(energy_maps, dim=0).mean(dim=0) # (B, C, H, W)
        
        # Calculate attention weights. Lower energy means higher attention.
        # We use sigmoid for a smoother attention map [0, 1] instead of 1/e,
        # which can be numerically unstable if energy is close to 0.
        # This is a common practice in modern attention mechanisms.
        attention_map = torch.sigmoid(1 - fused_energy) # (B, C, H, W)

        # Apply attention to the input feature map
        return x * attention_map

# --- 示例：如何将其集成到网络中 ---

class ResNetBlockWithCAMSAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlockWithCAMSAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # The CAMS-AM module
        self.camsam = CAMSAM(channels=out_channels, kernel_sizes=[3, 7, 11])
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply CAMS-AM after the main convolutions, before the residual connection.
        out = self.camsam(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# --- 测试代码 ---
if __name__ == '__main__':
    # Create a dummy input tensor
    input_tensor = torch.randn(4, 64, 32, 32) # (B, C, H, W)

    # Instantiate the ResNet-like block with CAMS-AM
    block = ResNetBlockWithCAMSAM(in_channels=64, out_channels=64)
    
    # Forward pass
    output_tensor = block(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    
    # Check if a standalone CAMSAM module works
    camsam_module = CAMSAM(channels=64)
    standalone_output = camsam_module(input_tensor)
    print("Standalone CAMSAM output shape:", standalone_output.shape)

