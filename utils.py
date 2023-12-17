from dataset import AnimeFacesDataset
import os
import pandas as pd
from torch.utils.data import DataLoader
import json


def create_indexes(config):
    index = list(os.listdir(config["data"]["data_path"]))

    # Чтение списка из файла с помощью json
    with open('my_list.json', 'r') as file:
        read_list = json.load(file)


    i = 0
    for part in config["data"]["parts"]:
        part_index = index[i:config["data"]["parts"][part]["limit"]]
        i = i+config["data"]["parts"][part]["limit"]

        with open(os.path.join(config["data"]["index_path"], f"{part}_{config['data']['parts'][part]['limit']}.json"), "w") as f: 
            json.dump(part_index, file)
    

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