import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import misc

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_channels

    def forward(self, x):
        return self.backbone(x)

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str = 'resnet18', 
                 train_backbone: bool = True,
                 return_interm_layers: bool = False,
                 dilation: bool = False):
        
        # Load ResNet18
        backbone = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=misc.FrozenBatchNorm2d)
        
        # For ResNet18, we'll use the features up to layer4
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        
        # Only keep the backbone layers we need
        layers_to_keep = ['conv1', 'bn1', 'relu', 'maxpool', 
                         'layer1', 'layer2', 'layer3', 'layer4']
        
        # Create new sequential model with just the layers we want
        backbone_layers = []
        for name, module in backbone.named_children():
            if name in layers_to_keep:
                backbone_layers.append(module)
        
        backbone = nn.Sequential(*backbone_layers)
        
        super().__init__(backbone, num_channels)
        
        # Freeze layers if not training backbone
        if not train_backbone:
            for name, parameter in self.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)

def build_backbone(args):
    backbone = Backbone(
        name='resnet18',
        train_backbone=args.lr_backbone > 0,
        return_interm_layers=False,
        dilation=False
    )
    return backbone
