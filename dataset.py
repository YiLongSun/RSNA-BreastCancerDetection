import cv2
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
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        # image
        image = cv2.imread(self.dataframe["path"][index])
        laterality = self.dataframe["laterality"][index]
        if (laterality=="L"):
            image = image[0:1024, 0:683]
        elif (laterality=="R"):
            image = image[0:1024, 342:1024]
        else:
            pass
        image = cv2.resize(image, (512, 512))
        image = (image * 255).astype(np.uint8)
        image = self.transform(image)
        
        # label
        label = self.dataframe["cancer"][index]

        return image, label