import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
        dataloader = combined_dataloader
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    all_labels = []
    all_preds = []
    differences = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float)
            outputs = model(images)
            preds = torch.round(torch.sigmoid(outputs))
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            differences.extend((preds - labels).cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    differences = np.array(differences)

    # Calculate mean and standard deviation of differences
    mean_diff = np.mean(differences, axis=0)
    std_diff = np.std(differences, axis=0)
    
    # Plot accuracy with error bars
    plt.figure()
    plt.errorbar(range(len(mean_diff)), mean_diff, yerr=std_diff, fmt='o', ecolor='blue', capsize=5)
    plt.plot([min(range(len(mean_diff))), max(range(len(mean_diff)))], [0, 0], 'k--')
    plt.xlabel('Batch index')
    plt.ylabel('Difference (Predicted - Actual)')
    plt.title('Model Accuracy with Error Bars')
    plt.savefig('accuracy_with_error_bars.png')
    plt.show()
    
    print(f"Accuracy plot with error bars has been saved as 'accuracy_with_error_bars.png'.")

if __name__ == "__main__":
    data_paths = [
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set1/testing_data/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data"
    ]
    test_model(data_paths)
