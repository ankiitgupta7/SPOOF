import os
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# === CONFIG ===
IMAGE_PATH = 'sampled_1000_images_from5000/ILSVRC2012_val_00013542.JPEG'

# === DEVICE SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD MODEL ===
model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
model.eval()

# === TRANSFORM ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# === LOAD IMAGE ===
image = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(image).unsqueeze(0).to(device)

print(f"Image shape: {img_tensor.shape}")
print("Model loaded and ready.")
