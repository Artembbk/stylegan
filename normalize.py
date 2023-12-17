import torch

def normalize_negative_one(img):
    normalized_input = (img - torch.amin(img)) / (torch.amax(img) - torch.amin(img))
    return 2*normalized_input - 1

def denormalize_from_negative_one(self, normalized_img):
    denormalized_input = 0.5 * (normalized_img + 1.0)
    denormalized_input = denormalized_input * 255.0
    return denormalized_input.type(torch.uint8)