import torch
import torch.nn as nn
from torch import Tensor

class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        # Classification head
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        
        # Bounding box regression head
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Input projection for backbone features
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: Tensor):
        # Extract features from backbone
        features = self.backbone(samples)
        
        # Project features to transformer dimension
        src = self.input_proj(features)
        
        # Create mask (assuming no padding for simplicity)
        mask = torch.zeros((src.shape[0], src.shape[2], src.shape[3]), 
                          dtype=torch.bool, device=src.device)
        
        # Forward through transformer
        hs = self.transformer(src, mask, self.query_embed.weight)
        
        # Process transformer outputs
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x