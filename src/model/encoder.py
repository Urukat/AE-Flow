import torch.nn as nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

## According to original paper, we use ImageNet-pretrained Wide ResNet-50-2 as the feature extractor
class Encoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        # Here we use default weights of pretrained wide_resnet50_2 which is equivalent to IMAGENET1K_V2 
        resnet = wide_resnet50_2(weights = Wide_ResNet50_2_Weights.DEFAULT)
        # By invesigating the wide_resnet50_2 model, we found we should discard the last 3 modules
        self.model = nn.Sequential(*list(resnet.children())[:-3])
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out