import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingSine(nn.Module):
    """
    Standard sinusoidal positional embedding used in DETR.
    This matches the implementation in the original DETR paper.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), 
                            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), 
                            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class DETRTransformer(nn.Module):
    """
    Simplified DETR transformer with ResNet18 compatibility
    """
    def __init__(self, d_model=512, nhead=8, num_queries=100, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Positional embedding
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        
        # Query embeddings (learnable)
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection for decoder queries
        self.decoder_norm = nn.LayerNorm(d_model)
        
    def forward(self, src, mask):
        # src: [batch_size, 512 (d_model), height, width] from ResNet18
        # mask: [batch_size, height, width] (True for padding areas)
        
        # Generate positional embeddings
        pos_embed = self.pos_embed(src, mask)  # [batch_size, d_model, h, w]
        
        # Forward through transformer
        hs, memory = self.transformer(
            src=src,
            mask=mask,
            query_embed=self.query_embed.weight,
            pos_embed=pos_embed
        )
        
        # Normalize decoder output
        hs = self.decoder_norm(hs)
        
        return hs, memory

def build_transformer(args):
    return DETRTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_queries=args.num_queries,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
