import math
import torch
import torch.nn as nn
import numpy as np
from nflows import distributions
from torchmetrics.functional import structural_similarity_index_measure

from .flow import NormalizingFlow
from .encoder import Encoder
from .decoder import Decoder

class AE_FLOW(nn.Module):
    def __init__(self, subnet):
        super().__init__()
        # Here we use default weights of pretrained wide_resnet50_2 which is equivalent to IMAGENET1K_V2 
        self.encoder = Encoder()
        self.flow = NormalizingFlow(subnet)
        self.decoder = Decoder()

    def forward(self, img):
        self.latent_z = self.encoder(img)
        # [batch_size, 1024, 16, 16] [batch_size]
        self.z_hat, self.jaz = self.flow(self.latent_z)
        # [batch_size, 3, 256, 256]
        self.rec_img = self.decoder(self.z_hat)
        return self.rec_img, self.z_hat, self.jaz

    # from https://towardsdatascience.com/introduction-to-normalizing-flows-d002af262a4b
    # or we can use torch.distribution.Normal to calculate the flow_loss
    def flow_loss(self, bits_per_pixel=True):
        # ignore the batch_size
        shape = self.z_hat.shape[1:]
        log_z = distributions.StandardNormal(shape=shape).log_prob(self.z_hat)
        log_prob = log_z + self.jaz
        if(bits_per_pixel):
            log_prob /= (math.log(2) * np.prod(shape))
        return -log_prob.mean(), log_z

    def anomaly_score(self, beta, log_z, img):
        # experimental code
        # Option 1, use log_z_hat instead of p_z_hat
        shape = self.z_hat.shape[1:]
        # Sflow = -distributions.StandardNormal(shape=shape).log_prob(self.z_hat) / np.prod(shape)
        # Option 2, follow the original paper use p_z_hat 
        Sflow = -torch.exp(distributions.StandardNormal(shape=shape).log_prob(self.z_hat) / np.prod(shape))
        
        Srecon = -structural_similarity_index_measure(preds=self.rec_img, target=img, kernel_size=11, reduction='sum')        
        # print("Srecon: {}".format(Srecon))
        # print("Sflow: {}".format(Sflow) )
        return beta * Sflow.mean() + (1 - beta) * Srecon


        