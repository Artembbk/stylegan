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

class FakeDataset(Dataset):
    def __init__(self, data):
        super(FakeDataset, self).__init__()

        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"images": self.data[idx]}

