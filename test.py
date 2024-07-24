import os
import torch
from model import AutoFocusModel
from data_loader2 import combined_dataloader

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
        dataloader = combined_dataloader(data_paths, batch_size=batch_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            outputs = model(images)
            for output, label in zip(outputs, labels):
                print(f"Predicted: {output.item():.4f}, Actual: {label.item():.4f}")

if __name__ == "__main__":
    data_paths = [
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set1/testing_data/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data"
    ]
    test_model(data_paths)
