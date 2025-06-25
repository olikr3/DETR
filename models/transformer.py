import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=False  # Keeping False for compatibility with original DETR
            ),
            num_layers=num_encoder_layers
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=False  # Keeping False for compatibility with original DETR
            ),
            num_layers=num_decoder_layers
        )
        
        self.d_model = d_model
        self.nhead = nhead
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed, pos_embed):
        # ResNet18 backbone outputs [batch, channels, height, width]
        # Flatten spatial dimensions and rearrange for transformer
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [hw, bs, c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        # Prepare query embeddings
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, bs, c]
        mask = mask.flatten(1)  # [bs, hw]
        
        # Add positional encoding to source
        src = src + pos_embed
        
        # Transformer encoder
        memory = self.encoder(src, src_key_padding_mask=mask)
        
        # Transformer decoder
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, 
                          memory_key_padding_mask=mask,
                          tgt_key_padding_mask=None)
        
        # Reshape outputs
        hs = hs.transpose(1, 2)  # [num_queries, bs, c] -> [bs, num_queries, c]
        memory = memory.permute(1, 2, 0).view(bs, c, h, w)
        
        return hs, memory

def build_transformer(args):
    return SimplifiedTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
