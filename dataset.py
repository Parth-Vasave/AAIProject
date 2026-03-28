import os
import cv2
import random
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CycleGANDataset(Dataset):
    def __init__(self, data_root, name, category, size=256):
        self.data_path = os.path.join(data_root, name, category)
        self.img_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(int(size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

def reconstruct_color(img_tensor):
    """
    Reverse normalization: (tensor * 0.5 + 0.5) * 255
    Input: (C, H, W) tensor in range [-1, 1]
    Output: (H, W, C) numpy array in range [0, 255]
    """
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255
    return np.clip(img, 0, 255).astype(np.uint8)

# Helper function to load a single image for inference
def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize(size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)
