import torch
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['overripe', 'ripe', 'rotten', 'unripe']
model_path = 'best_banana_resnet18.pth'  # path to your saved model

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

    # Preprocess
    img_tensor = val_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()

    # Display results
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} ({confidence_score*100:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"Prediction: {predicted_class} | Confidence: {confidence_score*100:.2f}%")

# === Run Example ===
if __name__ == "__main__":
    image_path = r"rotten.jpg"  # change this to your image path
    predict_image(image_path)
