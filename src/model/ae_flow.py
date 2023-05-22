import torch.nn as nn
from .flow import NormalizingFlow
from .encoder import Encoder
from .decoder import Decoder

class AE_FLOW(nn.Module):
    def __init__(self):
        super().__init__()
        # Here we use default weights of pretrained wide_resnet50_2 which is equivalent to IMAGENET1K_V2 
        self.encoder = Encoder()
        self.flow = NormalizingFlow()
        self.decoder = Decoder()

    def forward(self, img):
        self.latent_z = self.encoder(img)
        # [batch_size, 1024, 16, 16] [batch_size]
        self.z_hat, self.jaz = self.flow(self.latent_z)
        # [batch_size, 3, 256, 256]
        self.rec_img = self.decoder(self.z_hat)
        return self.rec_img