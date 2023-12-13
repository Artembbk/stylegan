import argparse
import json
from utils import get_dataloaders

def main(config):
    dataloaders = get_dataloaders(config)


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