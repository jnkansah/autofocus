import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from PIL import Image, ImageFile

# To handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_valid_image_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.lower().endswith(valid_extensions)

class CustomDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.filepaths = []
        self.labels = []
        self.transform = transform
        for root_dir in root_dirs:
            for label, subdir in enumerate(sorted(os.listdir(root_dir))):
                subdir_path = os.path.join(root_dir, subdir)
                if os.path.isdir(subdir_path):
                    for fname in os.listdir(subdir_path):
                        if is_valid_image_file(fname):
                            self.filepaths.append(os.path.join(subdir_path, fname))
                            self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        img_path = self.filepaths[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_paths, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = CustomDataset(root_dirs=data_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

if __name__ == "__main__":
    data_paths = [
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set1/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/healthy/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/parasites/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data"
    ]
    try:
        dataloader = load_data(data_paths)
        
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        
        # Display a few images
        dataiter = iter(dataloader)
        images, labels = next(dataiter)
        
        # Show images
        imshow(torchvision.utils.make_grid(images[:4]))
        print(' '.join(f'{labels[j]}' for j in range(4)))
    except RuntimeError as e:
        print(e)
