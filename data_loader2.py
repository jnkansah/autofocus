import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, random_split

# Define the data transformation
# transform = transforms.Compose([
#     transforms.Resize((386, 516)),
#     transforms.ToTensor()
# ])

# Define your datasets
# dataset1 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set1/training_data', transform=transform)
# dataset2 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set2/healthy/training_data', transform=transform)
# dataset3 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set2/parasites/training_data', transform=transform)

# dataset4 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set1/testing_data/training_data', transform=transform)
# dataset5 = datasets.ImageFolder(root='/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data', transform=transform)

# Combine datasets
# combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])

# print(dataset1.class_to_idx)
# print(dataset2.class_to_idx)
# print(dataset3.class_to_idx)

def combined_dataloader(data_paths, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((386, 516)),
        transforms.ToTensor()
    ])
    
    datasets_list = []
    class_to_idx = {}

    for path in data_paths:
        dataset = datasets.ImageFolder(root=path, transform=transform)
        datasets_list.append(dataset)
        
        # Merge class_to_idx dictionaries
        for class_name, class_idx in dataset.class_to_idx.items():
            if class_name not in class_to_idx:
                class_to_idx[class_name] = class_idx
            else:
                # Adjust class indices if necessary
                max_idx = max(class_to_idx.values())
                class_to_idx[class_name] = max_idx + 1

    combined_dataset = ConcatDataset(datasets_list)

    total_size = len(combined_dataset)

    train_size, test_size = int(total_size*0.9), int(total_size*0.1)
    train_set, test_set = random_split(combined_dataset, [train_size, test_size])
    
    # Create a DataLoader for the combined dataset
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)
    return train_dataloader, test_dataloader, class_to_idx



train_dataloader, test_dataloader, class_to_idx = combined_dataloader([
    # list of paths
    '/Users/jayda-louise.nkansah/Desktop/autofocus/set1/training_data',
    '/Users/jayda-louise.nkansah/Desktop/autofocus/set2/healthy/training_data',
    '/Users/jayda-louise.nkansah/Desktop/autofocus/set2/parasites/training_data'
])

