import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self):
        self.model = torchvision.models
    def forward(self, x):
        out = self.model(x)
        return out