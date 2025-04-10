import torch
import torch.nn as nn


class ModulatedConv1DLayer(nn.Module):
    def __init__(self, config):
        super(ModulatedConv1DLayer, self).__init__()
        self.conv1d_layer = nn.Conv1d(
            in_channels=config.hidden_size,         
            out_channels=config.hidden_size,        
            kernel_size=1,  
            stride=1,
            padding=0,
            bias=True
        )  # 权重的形状: [config.hidden_size, config.hidden_size, kernel_size]
        
    def initialize_conv(self):
        nn.init.zeros_(self.conv1d_layer.weight)
        if self.conv1d_layer.bias is not None:
            nn.init.zeros_(self.conv1d_layer.bias)

    
    def forward(self, x, style, demodulate=False):
        """
        x: 输入张量，形状为 [B, T*H*W, config.hidden_size]
        style: 调制向量，形状为 [config.hidden_size]
        demodulate: 是否进行权重的归一化，默认为 False
        """
        x = x.permute(0, 2, 1)
        B, C, T_HW = x.shape
        
        # 获取卷积层的权重，形状为 [out_channels, in_channels, kernel_size]
        w = self.conv1d_layer.weight  # [config.hidden_size, config.hidden_size, 1]
        
        # 1. 将权重 w 进行调制: 逐样本进行调制
        # w: [1, out_channels, in_channels, kernel_size]  -> 加一个维度用于调制
        w = w.unsqueeze(0)  # [1, config.hidden_size, config.hidden_size, 1]
        
        # style: [config.hidden_size] -> [1, config.hidden_size, 1, 1]
        style = style.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 广播到适应权重的形状
        
        # 调制权重: 每个卷积核乘以同一个风格向量
        w = w * style  # [1, config.hidden_size, config.hidden_size, 1]
        
        # 2. 执行卷积
        # 将输入的 x 进行 reshape 以匹配卷积输入
        # x = x.reshape(1, -1, T_HW)  # [1, B * config.hidden_size, T*H*W]
        
        # 将权重 reshape 以匹配输入维度
        w = w.reshape(-1, C, 1)  # [config.hidden_size, config.hidden_size, 1]
        
        # 执行一维卷积
        x = nn.functional.conv1d(x, w, stride=1, padding=0, groups=1)
        # x = x.reshape(B, -1, T_HW)  # 恢复 batch 维度
        
        # 3. 权重归一化（如果需要）
        if demodulate:
            d = torch.rsqrt(w.pow(2).sum([2]) + 1e-8)  # 计算归一化系数
            x = x * d.unsqueeze(-1)  # 对输出进行归一化
        
        x = x.permute(0, 2, 1)   
        return x
