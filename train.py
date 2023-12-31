import argparse
import json
from utils import get_dataloaders
from model import Generator
from model import Discriminator
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
import wandb
from utils import weights_init
from trainer import Trainer

def main(config):
    dataloaders = get_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(**config["arch"]["Generator"]["args"]).to(device)
    discriminator = Discriminator(**config["arch"]["Discriminator"]["args"]).to(device)

    print(generator)
    print(discriminator)

    print(discriminator.__class__.__name__)

    # generator.apply(weights_init)
    # discriminator.apply(weights_init)

    optim_d = getattr(optim, config["optimizer"]["name"])(discriminator.parameters(), **config["optimizer"]["args"])
    optim_g = getattr(optim, config["optimizer"]["name"])(generator.parameters(), **config["optimizer"]["args"])

    criterion = nn.BCELoss()

    wandb.init(project="anime")

    trainer = Trainer(
        config,
        dataloaders,
        generator,
        discriminator,
        optim_d,
        optim_g,
        criterion,
        device
    )

    trainer.train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing config file path')
    parser.add_argument('--config', type=str, help='Path to the config JSON file')
    args = parser.parse_args()
    config_path = args.config

    if config_path:
        with open(config_path, 'r') as file:
            config_data = json.load(file)
        config = config_data
    else:
        print("No config file path provided.")

    main(config)