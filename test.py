#!/usr/bin/env python3

import torch
from model import AutoFocusModel
from data_loader import load_data

def test_model(data_paths, model_path="autofocus_model.pth", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoFocusModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataloader = load_data(data_paths, batch_size=batch_size)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            outputs = model(images)
            for output, label in zip(outputs, labels):
                print(f"Predicted: {output.item():.4f}, Actual: {label.item():.4f}")

if __name__ == "__main__":
    data_paths = [
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set1/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/healthy/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/parasites/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data"
    ]
    test_model(data_paths)
