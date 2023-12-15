from torch import nn
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
import os
import torchvision.transforms as transforms

class AnimeFacesDataset(Dataset):
    def __init__(self, config, index_file):
        super(AnimeFacesDataset, self).__init__()

        self.data_path = config["data"]["data_path"]
        self.img_labels = pd.read_csv(index_file)
        self.transform = transforms.Compose(
            [
                transforms.Normalize(0.5, 0.5)
            ])

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_labels.iloc[idx, 0
                                                                     ])
        image = read_image(img_path)
        image = torch.tensor(image, dtype=torch.float32)
        image = self.transform(image)
        return image

    