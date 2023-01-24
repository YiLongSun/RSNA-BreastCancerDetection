import cv2
import dicomsdl
from torch import nn
from torchvision.models import resnet50

class RSNA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = resnet50(pretrained=True)
        self.classification = nn.Linear(1000, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classification(x)
        return x
        