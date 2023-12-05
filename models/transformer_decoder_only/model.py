import torch
import torch.nn as nn

class TransformerDecoderOnly(nn.Module):
    # d_model : number of features
    def __init__(self, feature_size: int, in_seq_len: int, out_seq_len: int, n_head: int, Nx: int, d_ff: int = 1024, d_d: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = feature_size, nhead = n_head, dim_feedforward = d_ff, dropout = dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = Nx)        
        self.decoder1 = nn.Linear(feature_size*in_seq_len, d_d)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p = dropout)
        self.decoder2 = nn.Linear(d_d, out_seq_len)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # else:
            #     nn.init.zeros_(p)

    def forward(self, src, device) -> torch.Tensor:
        # comment to omit the causal mask
        #mask = torch.nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(device)
        encoder_out = self.transformer_encoder(src)
        out = self.decoder1(encoder_out.view(encoder_out.size(0), -1))
        out = self.relu(out)
        #out = self.dropout(out)
        output = self.decoder2(out)

        # encoder_out = self.transformer_encoder(src)
        # output = torch.relu(self.decoder1(encoder_out.view(encoder_out.size(0), -1)))
        # output = self.decoder2(output)

        return output