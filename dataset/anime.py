from torch import nn
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
import os

class AnimeFacesDataset(Dataset):
    def __init__(self, config, part, index_file):
        super(AnimeFacesDataset, self).__init__()

        self.data_path = config["dataset"][part]["path"]
        self.img_labels = pd.read_csv(index_file)
        self.img_labels = self.img_labels[:config["data"][part]["limit"]]

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        return image

    