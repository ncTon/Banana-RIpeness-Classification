import torch
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['overripe', 'ripe', 'rotten', 'unripe']
model_path = 'best_banana_resnet18.pth'  # path to saved model

# === Load Model ===
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_features, 4)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Define Transform ===
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Predict Function ===
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



# === Run Example ===
if __name__ == "__main__":
    image_path = r"Group_41_Banana_images\rotten.jpg"  # change this to image path
    predict_image(image_path)
