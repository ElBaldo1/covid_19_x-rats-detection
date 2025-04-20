import torch
import sys
from PIL import Image
from torchvision import transforms
from covid_classification import RadiographyCNN

# Define the same transform as used in training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = RadiographyCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load and preprocess image
if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    label = "COVID-positive" if prob[1] > prob[0] else "COVID-negative"
    confidence = prob.max().item() * 100

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")
