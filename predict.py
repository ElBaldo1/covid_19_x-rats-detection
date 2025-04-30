#!/usr/bin/env python
import time, psutil, platform, torch, glob, os
from PIL import Image
from torchvision import transforms
from covid_classification import RadiographyCNN

def predict_one(path, model, transform):
    img = Image.open(path).convert("RGB")
    x   = transform(img).unsqueeze(0)
    t0  = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    dt = (time.perf_counter() - t0) * 1000  # milliseconds
    prob  = torch.softmax(logits, 1)[0]
    label = "COVID-positive" if prob[1] > prob[0] else "COVID-negative"
    conf  = prob.max().item() * 100
    return label, conf, dt

# System information
print("System:", platform.platform())
print("CPU:", platform.processor() or "Apple M1")
print(f"RAM: {psutil.virtual_memory().total/2**30:.1f} GB")
print("-" * 50)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = RadiographyCNN()
model.load_state_dict(
    torch.load("best_model.pth", map_location="cpu", weights_only=True)
)
model.eval()
print("Model loaded successfully.\n")

# Locate images
images = sorted(glob.glob("COVID-Data-Radiography/manual-test/*.png"))

if not images:
    print("No PNG images found in COVID-Data-Radiography/manual-test/. Exiting.")
    exit()

print(f"Found {len(images)} images for inference.\n")

# Inference loop
for path in images:
    label, conf, ms = predict_one(path, model, transform)
    print(f"{os.path.basename(path):30}  →  {label:15}  ({conf:.2f} %)   [{ms:.1f} ms]")
