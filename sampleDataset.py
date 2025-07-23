'''import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Load MNIST image (grayscale, 1 channel)
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
img, label = mnist[0]  # Take one image
img = img.unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]

# Define a simple CNN to get the feature maps
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

# Create model and get feature maps
model = SimpleCNN()
with torch.no_grad():
    feature_maps = model(img)  # Shape: [1, 32, 28, 28]

# Remove batch dimension
feature_maps = feature_maps.squeeze(0)

# Plot all 32 feature maps
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i in range(32):
    ax = axes[i // 8, i % 8]
    ax.imshow(feature_maps[i].cpu(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'FM {i+1}', fontsize=8)

plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Load MNIST and get one image
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

img, label = mnist[0]  # Get one sample
print(f"Label: {label}")
print(f"Image shape: {img.shape}")  # [1, 28, 28]

# Show the original image
plt.imshow(img.squeeze(0), cmap='gray')
plt.title("Original MNIST Image")
plt.axis('off')
plt.show()  # <-- show here to avoid overwrite

# 2. Add batch dimension for Conv2D input
img = img.unsqueeze(0)  # [1, 1, 28, 28]

# 3. Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

# 4. Pass image through CNN
model = SimpleCNN()
with torch.no_grad():
    output = model(img)  # [1, 32, 28, 28]

# 5. Visualize all 32 feature maps
feature_maps = output.squeeze(0)  # [32, 28, 28]

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i in range(32):
    ax = axes[i // 8, i % 8]
    ax.imshow(feature_maps[i].cpu(), cmap='gray')
    ax.axis('off')
    ax.set_title(f"FM {i+1}", fontsize=8)

plt.suptitle("32 Feature Maps from Conv Layer", fontsize=14)
plt.tight_layout()
plt.show()
'''
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Load MNIST and get one image
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

img, label = mnist[0]  # Get one sample
print(f"Label: {label}")
print(f"Image shape: {img.shape}")  # [1, 28, 28]

# Show the original image
plt.imshow(img.squeeze(0), cmap='gray')
plt.title("Original MNIST Image")
plt.axis('off')
plt.show()  # Show original image first

# 2. Add batch dimension for Conv2D input
img = img.unsqueeze(0)  # [1, 1, 28, 28]

# 3. Define CNN with two conv layers: 1->32 channels, then 32->64 channels
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 1 input channel, 32 output
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 input channels, 64 output

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

# 4. Pass image through CNN
model = SimpleCNN()
with torch.no_grad():
    output = model(img)  # Output shape: [1, 64, 28, 28]

# 5. Visualize all 64 feature maps from the second conv layer
feature_maps = output.squeeze(0)  # [64, 28, 28]

fig, axes = plt.subplots(8, 8, figsize=(16, 16))
for i in range(64):
    ax = axes[i // 8, i % 8]
    ax.imshow(feature_maps[i].cpu(), cmap='gray')
    ax.axis('off')
    ax.set_title(f"FM {i+1}", fontsize=6)

plt.suptitle("64 Feature Maps from Second Conv Layer", fontsize=16)
plt.tight_layout()
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

# Create input values from -5 to 5
x = np.linspace(-5, 5, 50,-100,200)

# Compute ReLU output y = max(0, x)
y = np.maximum(0, x)

# Plot
plt.plot(x, y, label='ReLU: y = max(0, x)')
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.title("ReLU Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.show()
'''
