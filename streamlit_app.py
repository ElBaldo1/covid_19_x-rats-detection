import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from covid_classification import RadiographyCNN

# Page config must be the FIRST Streamlit command
st.set_page_config(page_title="COVID-19 X-ray Classifier", layout="centered")

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model (CPU-only for Streamlit)
@st.cache_resource
def load_model():
    model = RadiographyCNN()
    model.load_state_dict(torch.load(
        "best_model.pth",
        map_location="cpu",
        weights_only=True
    ))
    model.eval()
    return model

model = load_model()
class_mapping = {0: "COVID-negative", 1: "COVID-positive"}

# App layout
st.title("COVID-19 Chest X-ray Classifier")
st.markdown("Upload a chest X-ray image (JPG or PNG) to classify it as **COVID-positive** or **COVID-negative**.")

uploaded = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            label = class_mapping[pred_idx]
            confidence = probs[pred_idx].item() * 100

    st.success(f"**Prediction:** {label}")
    st.metric(label="Confidence", value=f"{confidence:.2f}%")
