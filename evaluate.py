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

test_data = ImageFolder('Banana Ripeness Classification Dataset/test', transform=val_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# === 3. Evaluate ===
all_preds, all_labels = [], []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

# === 4. Metrics ===
test_accuracy = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f'\nTest Accuracy: {test_accuracy:.2f}%')

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
# Single image prediction example
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Show original image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Preprocess and predict
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    # Plot prediction probabilities
    plt.subplot(1, 2, 2)
    probs = probabilities.cpu().numpy()[0]
    plt.bar(class_names, probs)
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence_score:.2f}')
    plt.xticks(rotation=45)
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence_score

