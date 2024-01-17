
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from patch import read_slide_patches

class SlideDataset(Dataset):
    def __init__(self, slide_paths, transform=None):
        self.slide_paths = slide_paths
        self.transform = transform

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        slide_path = self.slide_paths[idx]
        patches = read_slide_patches(slide_path)
        return torch.stack([self.transform(patch) for patch in patches])
