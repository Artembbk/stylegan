import argparse
import json
from utils import get_dataloaders
import matplotlib.pyplot as plt

def main(config):
    dataloaders = get_dataloaders(config)
    train_dataloader = dataloaders['train']

    train_features = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    plt.imshow(img)
    plt.show()

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