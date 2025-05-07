import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import urllib.request

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

# === LOAD IMAGENET CLASS LABELS ===
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# === RUN INFERENCE ===
with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    top5_probs, top5_idxs = probabilities.topk(5)

# === DISPLAY TOP-5 ===
top5_probs = top5_probs.squeeze(0).tolist()
top5_idxs = top5_idxs.squeeze(0).tolist()

print("\nTop-5 Predictions:")
for rank, (idx, prob) in enumerate(zip(top5_idxs, top5_probs), start=1):
    label = imagenet_classes[idx]
    print(f"{rank}: {label} (class {idx}) - confidence: {prob:.4f}")
