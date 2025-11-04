import torch
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# === 1. Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_features, 4)
)
model.load_state_dict(torch.load('best_banana_resnet18.pth', map_location=device))
model.to(device)
model.eval()

# === 2. Data setup ===
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets
train_data = ImageFolder('Banana Ripeness Classification Dataset/train', transform=val_transform)
valid_data = ImageFolder('Banana Ripeness Classification Dataset/valid', transform=val_transform)
test_data  = ImageFolder('Banana Ripeness Classification Dataset/test', transform=val_transform)

# Loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

# === 3. Helper function ===
def evaluate(loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    acc = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, all_preds, all_labels

# === 4. Evaluate all ===
train_acc, _, _ = evaluate(train_loader)
valid_acc, _, _ = evaluate(valid_loader)
test_acc, all_preds, all_labels = evaluate(test_loader)

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {valid_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# === 5. Confusion Matrix & Report ===
class_names = ['rotten', 'ripe', 'overripe', 'unripe']

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
