
import torch
import torch.nn as nn

class AF2Transformer(nn.Module):

    def __init__(self,
                 input_dim=128,
                 hidden_dim=256,
                 num_heads=8,
                 num_layers=4):

        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        # x shape:
        # (B,L,L,C)

        x = x.mean(dim=2)

        x = self.input_proj(x)

        x = self.transformer(x)

        x = self.classifier(x)

        return torch.sigmoid(x).squeeze(-1)
