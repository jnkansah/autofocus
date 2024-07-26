import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import AutoFocusModel
from data_loader2 import test_dataloader, class_to_idx

from tqdm import tqdm

def test_model(data_paths, model_path="autofocus_model.pth", batch_size=32):
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoFocusModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    print("Model loaded and set to evaluation mode.")
    
    try:
        dataloader = test_dataloader
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    all_labels = []
    all_preds = []

    counter = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            outputs = model(images)
            
            

            outputs = outputs.tolist()

            all_preds.extend(outputs)

            all_labels.extend(labels.tolist())

            counter += 1

            if counter > 19:
                break
    
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    print(idx_to_class)
    
    all_preds = [int(a[0]) for a in all_preds]

    print(all_preds)

    new_all_preds = [idx_to_class[prediction] for prediction in all_preds]

    print(new_all_preds)

    all_labels_names = [idx_to_class[int(label)] for label in all_labels]

    # FUN WORK
    # 1. Flip the class_to_idx map --> done
    # 2. Map all the predictions from their idx, to the actual label
    # 3. Convert those labels into numbers


    plt.plot(all_labels_names,new_all_preds,"o", markersize=4,mec="black", alpha=0.7)
    plt.plot(all_labels_names,all_labels_names, "--", color="black")
    plt.show()
    
    
    # print(f"Accuracy plot with error bars has been saved as 'accuracy_with_error_bars.png'.")

if __name__ == "__main__":
    data_paths = [
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set1/testing_data/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data"
    ]
    test_model(data_paths)
