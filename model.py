from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class RSNA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 1)
        self.out = nn.Sigmoid()

    
    def forward(self, x):
        x = self.model(x)
        x = self.out(x)
        return x
        