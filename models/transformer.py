import torch
import torch.nn as nn

class AF2BindTransformer(nn.Module):

    def __init__(
        self,
        input_dim=2560,
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # x: (batch, L, 2560)

        x = self.input_proj(x)

        x = self.transformer(x)

        out = self.classifier(x)

        return out.squeeze(-1)
