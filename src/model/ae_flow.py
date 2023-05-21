import torch.nn as nn
from .encoder import Encoder

class AE_FLOW(nn.Module):
    def __init__(self):
        super().__init__()
        # Here we use default weights of pretrained wide_resnet50_2 which is equivalent to IMAGENET1K_V2 
        self.encoder = Encoder()
        print(self.encoder)

    def forward(self, x):
        self.latent_z = self.encoder(x)
        return self.latent_z