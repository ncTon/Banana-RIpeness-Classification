import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import Counter
from torchvision.models import resnet18, ResNet18_Weights

# Use pretrained weights safely (new syntax)
model = resnet18(weights=ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 4)
)

print("Using ResNet18 with pretrained weights")



#Import files
base_path = "Banana Ripeness Classification Dataset"
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")
val_path = os.path.join(base_path, "valid")
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class BananaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['overripe', 'ripe', 'rotten', 'unripe']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
train_dataset = BananaDataset(train_path, transform=train_transform)
test_dataset = BananaDataset(test_path, transform=val_transform)
val_dataset = BananaDataset(val_path, transform=val_transform)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
train_counts = Counter([train_dataset.classes[label] for label in train_dataset.labels])
val_counts = Counter([val_dataset.classes[label] for label in val_dataset.labels])
test_counts = Counter([test_dataset.classes[label] for label in test_dataset.labels])
print("\nTrain Class Distribution:")
for class_name, count in train_counts.items():
    print(f"  {class_name}: {count}")
print("\nValidation Class Distribution:")
for class_name, count in val_counts.items():
    print(f"  {class_name}: {count}")
print("\nTest Class Distribution:")
for class_name, count in test_counts.items():
    print(f"  {class_name}: {count}")



# Create transfer learning model - ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 4)
)


print("Using ResNet18 with pretrained weights")


criterion = nn.CrossEntropyLoss()

# Different learning rates for backbone and classifier
backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
classifier_params = model.fc.parameters()

optimizer = optim.Adam([
    {'params': backbone_params, 'lr': 0.0001},  # Lower LR for pre-trained features
    {'params': classifier_params, 'lr': 0.001}  # Higher LR for new classifier
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0

# Training parameters
num_epochs = 10

print(f"\nStarting training for {num_epochs} epochs...")

# Training loop
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total


#Evaluating the model
model.eval()
val_loss = 0.0
correct = 0
total = 0
    
with torch.no_grad():
        for data, target in val_loader:
            
            outputs = model(data)
            val_loss += criterion(outputs, target).item()
            
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
val_loss /= len(val_loader)
val_acc = 100. * correct / total
    
    # Update learning rate
scheduler.step()
    
    # Store metrics
train_losses.append(train_loss)
train_accuracies.append(train_acc)
val_losses.append(val_loss)
val_accuracies.append(val_acc)
    
print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save best model
if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_banana_resnet18.pth')
        print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
