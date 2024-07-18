import torch
import torch.optim as optim
import torch.nn as nn
from model import AutoFocusModel
from data_loader import load_data

def train_model(data_paths, epochs=192, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoFocusModel().to(device)
    dataloader = load_data(data_paths, batch_size=batch_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
    
    torch.save(model.state_dict(), "autofocus_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    data_paths = [
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set1/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/healthy/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/parasites/training_data",
        "/Users/jayda-louise.nkansah/Desktop/autofocus/set2/testing/training_data"
    ]
    train_model(data_paths)
