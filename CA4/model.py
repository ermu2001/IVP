import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, apply_to_dim_idx=-1):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.apply_to_dim_idx = apply_to_dim_idx

    def forward(self, x):
        variance = x.pow(2).mean(dim=self.apply_to_dim_idx, keepdim=True) + self.eps
        
        expand_slice = [slice(None)] + [None] * ((x.ndim - self.apply_to_dim_idx) \
                                          if self.apply_to_dim_idx >= 0 else \
                                          -self.apply_to_dim_idx - 1)
        weight = self.weight[expand_slice]  # shape: (1, 1, ..., dim)
        return weight * x / variance.sqrt()

class RetainedShapeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(RetainedShapeConvBlock, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for same spatial shape"
        stride = 1
        padding = kernel_size // 2
        self.norm = RMSNorm(in_channels, apply_to_dim_idx=-3)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(self.norm(x)))
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, spatial_scale_factor):
        super(DownsampleBlock, self).__init__()
        self.conv1 = RetainedShapeConvBlock(in_channels, out_channels, kernel_size)
        self.conv2 = RetainedShapeConvBlock(out_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(spatial_scale_factor)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, spatial_scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = lambda x: F.interpolate(x, scale_factor=spatial_scale_factor, mode='bilinear')
        self.conv1 = RetainedShapeConvBlock(in_channels, in_channels, kernel_size)
        self.conv2 = RetainedShapeConvBlock(in_channels, out_channels, kernel_size)
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            main_channel,
            depth,
            spatial_scale_factor=2
        ):
        super(UNet, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, main_channel, kernel_size=1)
        self.unet_downblocks = nn.ModuleList()
        self.unet_midblocks = nn.ModuleList()
        self.unet_upblocks = nn.ModuleList()
        channel_scale_factor = spatial_scale_factor ** 2
        for i in range(depth):
            self.unet_downblocks.append(DownsampleBlock(main_channel, main_channel * channel_scale_factor, kernel_size, spatial_scale_factor))
            main_channel *= channel_scale_factor
        
        self.unet_midblocks.extend([
            RetainedShapeConvBlock(main_channel, main_channel, kernel_size),
            RetainedShapeConvBlock(main_channel, main_channel, kernel_size),
        ])


        for i in range(depth):
            self.unet_upblocks.append(UpsampleBlock(main_channel * 2, main_channel // channel_scale_factor, kernel_size, spatial_scale_factor))
            main_channel //= channel_scale_factor

        self.final_conv = nn.Conv2d(main_channel, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.in_conv(x)
        for downblock in self.unet_downblocks:
            x = downblock(x)
            skip_connections.append(x)
        for midblock in self.unet_midblocks:
            x = midblock(x)
        for upblock in self.unet_upblocks:
            skip = skip_connections.pop()
            x = torch.concat((x, skip), dim=1)
            x = upblock(x)

        return self.final_conv(x)
    

if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, kernel_size=3, main_channel=4, depth=2, spatial_scale_factor=2).cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    y = model(x)
    print(y.shape)  # Should output torch.Size([1, 1, 256, 256]) or similar depending on the output channels