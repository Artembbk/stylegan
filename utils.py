from dataset import AnimeFacesDataset
import os
import pandas as pd
from torch.utils.data import DataLoader
import json
import torch


def create_indexes(config):
    index = list(os.listdir(config["data"]["data_path"]))

    i = 0
    for part in config["data"]["parts"]:
        part_index = index[i:i+config["data"]["parts"][part]["limit"]]
        i = i+config["data"]["parts"][part]["limit"]
        

        with open(os.path.join(config["data"]["index_path"], f"{part}_{config['data']['parts'][part]['limit']}.json"), "w") as f: 
            json.dump(part_index, f)
    

def get_dataloaders(config):
    dataloaders = {}
    for part in config["data"]["parts"]:
        index_path = os.path.join(config["data"]["index_path"], f"{part}_{config['data']['parts'][part]['limit']}.json")
        if not os.path.exists(index_path):
            create_indexes(config)
        dataset = AnimeFacesDataset(config, index_path)
        dataloaders[part] = DataLoader(dataset, batch_size=config["data"]["parts"][part]["batch_size"], shuffle=True if part == "train" else False)
    return dataloaders

def get_padding_t(stride, kernel_size):
    # Вычисляем паддинг для увеличения размера в 2 раза
    padding = (kernel_size - 2) // 2 + (stride - 1)
    return padding

def normalize_negative_one(self, img):
    normalized_input = (img - torch.amin(img)) / (torch.amax(img) - torch.amin(img))
    return 2*normalized_input - 1

def denormalize_from_negative_one(self, normalized_img):
    denormalized_input = 0.5 * (normalized_img + 1.0)
    denormalized_input = denormalized_input * 255.0
    return denormalized_input.type(torch.uint8)