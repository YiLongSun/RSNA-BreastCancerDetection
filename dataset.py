import cv2
import dicomsdl
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class RSNA_Dataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.dataframe, self.is_train = dataframe, is_train
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        # image
        image = cv2.imread(self.dataframe["path"][index])
        image = cv2.resize(image, (224, 224))
        image = (image * 255).astype(np.uint8)
        image = self.transform(image)
        
        # label
        label = self.dataframe["cancer"][index]

        return image, label