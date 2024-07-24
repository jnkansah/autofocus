import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader

# Define the data transformation
transform = transforms.Compose([
    transforms.Resize((386, 516)),
    transforms.ToTensor()
])

# Define your datasets
dataset1 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set1/training_data', transform=transform)
dataset2 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set2/healthy/training_data', transform=transform)
dataset3 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set2/parasites/training_data', transform=transform)

# Combine datasets
combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])

# Create a DataLoader for the combined dataset
combined_dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)




