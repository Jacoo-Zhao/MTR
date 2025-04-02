import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class MTRT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EEGEncoder()
        self.decoder = TrajectoryDecoder()

    def forward(self, eeg):
        memory = self.encoder(eeg)  # [B, T1, D]
        tgt = torch.zeros(eeg.size(0), 60, 256).to(eeg.device)  # 初始化查询序列
        return self.decoder(memory, tgt)  # [B, 60, 18]
    
class TrajectoryDecoder(nn.Module):
    def __init__(self, model_dim=256, n_heads=8, num_layers=6, output_dim=18):
        super().__init__()
        self.pos_encoding = PositionalEncoding(model_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=n_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)
        self.lstm = nn.LSTM(input_size=model_dim, hidden_size=model_dim, num_layers=1, batch_first=True)

    def forward(self, memory, tgt):  # memory: [B, T1, D], tgt: [B, T2, D]
        tgt = self.pos_encoding(tgt)
        tgt, _ = self.lstm(tgt)
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        out = self.transformer(tgt, memory).permute(1, 0, 2)
        return self.output_layer(out)  # [B, T2, output_dim]

    # def forward(self, memory, tgt, max_len=60):  # memory: [B, T1, D], tgt: [B, 1, D]
    #     B, _, D = memory.size()
    #     outputs = []
    #     tgt_step = torch.zeros(B, 1, D).to(memory.device)

    #     for t in range(max_len):
    #         tgt_step = self.pos_encoding(tgt_step)
    #         out = self.transformer(tgt_step, memory)
    #         pred = self.output_layer(out[:, -1:, :])  # 只取最后一步
    #         outputs.append(pred)
    #         tgt_step = torch.cat([tgt_step, pred], dim=1)

        return torch.cat(outputs, dim=1)  # [B, 60, 18]
    
class EEGEncoder(nn.Module):
    def __init__(self, input_dim=50, model_dim=256, n_heads=8, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, T=3600, C=50]
        x = self.input_proj(x)  # [B, T, D]
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)  # for transformer: [T, B, D]
        return self.transformer(x).permute(1, 0, 2)  # [B, T, D]
    

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)