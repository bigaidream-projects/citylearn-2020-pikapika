from torch import nn


class LinearDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super().__init__()
        self.tgt_decoder = nn.Linear(hidden_size, target_size)

    def forward(self, dec):
        return self.tgt_decoder(dec)
