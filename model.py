import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoFocusModel(nn.Module):
    def __init__(self):
        super(AutoFocusModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the feature map after conv and pooling layers
        # Input size: (3, 386, 516)
        # After conv1: (16, 386, 516)
        # After max_pool2d (2x2): (16, 193, 258)
        # After conv2: (32, 193, 258)
        # After max_pool2d (2x2): (32, 96, 129)
        
        self.fc1 = nn.Linear(32 * 96 * 129, 128)  # Adjusted size
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = AutoFocusModel()
    print(model)
