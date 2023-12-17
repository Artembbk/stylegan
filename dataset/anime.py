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
import json

class AnimeFacesDataset(Dataset):
    def __init__(self, config, index_file):
        super(AnimeFacesDataset, self).__init__()

        self.data_path = config["data"]["data_path"]
        with open(index_file, 'r') as file:
            self.img_labels = json.load(file)

    def normalize_negative_one(self, img):
        normalized_input = (img - torch.amin(img)) / (torch.amax(img) - torch.amin(img))
        return 2*normalized_input - 1

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_labels[idx])
        image = read_image(img_path)
        image = torch.tensor(image, dtype=torch.float32)
        image = self.normalize_negative_one(image)
        return {"images": image}

