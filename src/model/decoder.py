import torch
import torch.nn as nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.build_block(1024, 512)
        self.block2 = self.build_block(512, 256)
        self.block3 = self.build_block(256, 64)
        self.block4 = self.build_block(64, 3, last_layer=True)

        self.model = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4
        )

    def forward(self, x):
        rec_img = self.model(x)
        return rec_img

    # for the last layer we directly output the reconstructed image which means channel our is 3
    def build_block(self, channels_in, channels_out, last_layer=True):
        if(last_layer):
            return nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, kernel_size=(2, 2), stride=2),
                nn.Tanh(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, kernel_size=(2, 2), stride=2),
                nn.ReLU(),
                nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU()
            )