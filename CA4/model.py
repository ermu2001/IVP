import os
import os.path as osp
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
import warnings


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

def upsample(x, spatial_scale_factor):
    # this function couldn't be performed in bfloat16
    # there is no deterministic way to do interpolation
    return F.interpolate(x.float(), scale_factor=spatial_scale_factor, mode='bilinear').to(dtype=x.dtype)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, spatial_scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = functools.partial(upsample, spatial_scale_factor=spatial_scale_factor)
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
            spatial_scale_factor=2,
            channel_scale_factor=4,
        ):
        super(UNet, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, main_channel, kernel_size=1)
        self.unet_downblocks = nn.ModuleList()
        self.unet_midblocks = nn.ModuleList()
        self.unet_upblocks = nn.ModuleList()
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

def save_model(model_dir, model_cfg: DictConfig, model):
    """Save the model to the specified path."""
    if osp.exists(model_dir):
        # raise ValueError(f"Model directory {model_dir} already exists!")
        warnings.warn(f"Model directory {model_dir} already exists! Overwriting.")
    else:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), osp.join(model_dir, "model.pth"))
    with open(osp.join(model_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(model_cfg))
        
def load_model(model_dir=None, model_cfg: DictConfig=None):
    """Load the model from the specified path."""
    if model_cfg is None:
        model_cfg = OmegaConf.load(osp.join(model_dir, "config.yaml"))


    dtype = getattr(torch, model_cfg.dtype) if "dtype" in model_cfg else torch.float32
    model = UNet(**model_cfg).to(dtype=dtype)
    if model_dir is not None:
        model.load_state_dict(torch.load(osp.join(model_dir, "model.pth")))
    return model


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1, kernel_size=3, main_channel=4, depth=2, spatial_scale_factor=2).cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    y = model(x)
    print(y.shape)  # Should output torch.Size([1, 1, 256, 256]) or similar depending on the output channels