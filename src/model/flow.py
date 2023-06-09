import torch
import torch.nn as nn
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock 
from FrEIA.modules import InvertibleModule 
class NormalizingFlow(nn.Module):
    def __init__(self, subnet="resnet_type", flow_step=8):
        super().__init__()
        # According to the original paper, here we implement two types of subnet: conv_type and resnet_type
        self.inn = SequenceINN(1024, 16, 16)
        for k in range(flow_step):
            if(subnet == "conv_type"):
                self.inn.append(AllInOneBlock, subnet_constructor=subnet_conv)
            elif(subnet == "resnet_type"):
                # raise NotImplementedError
                self.inn.append(AllInOneBlock, subnet_constructor=subnet_restnet)
            else:
                raise NotImplementedError

    def forward(self, x):
        z_hat, jaz = self.inn(x)
        return z_hat, jaz

# for the intermiditae channel it is unknown, so here we fixed it as dims_in
def subnet_conv(dims_in, dims_out):
    return nn.Sequential(
       nn.Conv2d(dims_in, dims_in, kernel_size=(3, 3), padding=1),
       nn.ReLU(),
       nn.Conv2d(dims_in, dims_out, kernel_size=(1, 1)),
    )

class Resnet_flow(nn.Module):
    def __init__(self, dims_in, dims_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dims_in, dims_out, (3, 3), padding=1),
            nn.BatchNorm2d(dims_out),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(dims_in, dims_out, (1, 1))
        )
                
    def forward(self, x):
        out = self.layers(x)
        short_out = self.shortcut(x)
        return out + short_out

# here we make a new class resnet_type, which functionally same with nn.Sequential
def subnet_restnet(dims_in, dims_out):
    return Resnet_flow(dims_in, dims_out)
