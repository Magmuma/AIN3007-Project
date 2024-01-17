
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from training import train_model
from dataset import SlideDataset
from model import CNN
from patch import read_slide_patches, split_data

OPENSLIDE_PATH = r'C:\Users\Acer\miniconda3\envs\UniProject\Lib\site-packages\openslide\openslide-win64-20231011\bin'

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def main():
    data_folder = 'AIN3007_project/train_valid_data'

    # Split data into training and validation sets
    image_train, image_valid, _, _ = split_data(data_folder)

    print(f"Training set size: {len(image_train)}")
    print(f"Validation set size: {len(image_valid)}")

    # Create datasets and dataloaders
    train_dataset = SlideDataset(image_train, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Create the model, loss function, and optimizer
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=5)

    # Example: Read patches from the first validation slide and predict masks
    if image_valid:
        slide_path = image_valid[0]
        patches_valid = read_slide_patches(slide_path)
        tensor_patches = torch.stack([ToTensor()(patch) for patch in patches_valid])
        predicted_masks = model(tensor_patches)

if __name__ == "__main__":
    main()
