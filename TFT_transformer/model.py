import torch
import torch.nn as nn

class TFTModel(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()

        # embeddings
        self.day_emb = nn.Embedding(7, 8)
        self.menu_emb = nn.Embedding(5, 8)
        #projection
        self.input_proj = nn.Linear(1, d_model)

        # i thing this is encoder part of the transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # attention p
        self.attn = nn.Linear(d_model, 1)

        # final layers
        self.fc = nn.Sequential(
            nn.Linear(d_model + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, day, menu):
        # x_seq: (B, T, 1)

        x = self.input_proj(x_seq)  # (B, T, d_model)
        x = self.transformer(x)     # (B, T, d_model)

        # attention pooling
        attn_weights = torch.softmax(self.attn(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)  # (B, d_model)

        # embeddings
        day_e = self.day_emb(day)
        menu_e = self.menu_emb(menu)

        # THIS IS were the fussin accures so we called it fussion 
        x = torch.cat([x, day_e, menu_e], dim=1)

        return self.fc(x)