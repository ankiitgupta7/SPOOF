import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import urllib.request

# === CONFIG ===
IMAGE_PATH = 'sampled_1000_images_from5000/ILSVRC2012_val_00013542.JPEG'
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# === DEVICE SETUP ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD MODEL ===
model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
model.eval()

# === TRANSFORMS ===
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])
unnormalize = T.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

# === LOAD IMAGE ===
image = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(image).unsqueeze(0).to(device)
print(f"Image shape: {img_tensor.shape}")

# === LOAD IMAGENET LABELS ===
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# === ORIGINAL PREDICTION ===
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    top_probs, top_idxs = probs.topk(5)

top_probs = top_probs.squeeze(0).tolist()
top_idxs = top_idxs.squeeze(0).tolist()
true_top1_idx = top_idxs[0]
true_top1_label = imagenet_classes[true_top1_idx]

print("\nOriginal Top-5 Predictions:")
for rank, (idx, prob) in enumerate(zip(top_idxs, top_probs), start=1):
    print(f"{rank}: {imagenet_classes[idx]} (class {idx}) - confidence: {prob:.4f}")

# === ONE-PIXEL ATTACK ===
modified = img_tensor.clone()
_, _, h, w = modified.shape
x = random.randint(0, w - 1)
y = random.randint(0, h - 1)
ch = random.randint(0, 2)
original_val = modified[0, ch, y, x].item()
modified[0, ch, y, x] = torch.rand(1).item()

# === NEW PREDICTION ===
with torch.no_grad():
    output_adv = model(modified)
    probs_adv = F.softmax(output_adv, dim=1)
    top_probs_adv, top_idxs_adv = probs_adv.topk(5)

top_probs_adv = top_probs_adv.squeeze(0).tolist()
top_idxs_adv = top_idxs_adv.squeeze(0).tolist()
adv_top1_idx = top_idxs_adv[0]
adv_top1_label = imagenet_classes[adv_top1_idx]

print("\nModified Top-5 Predictions:")
for rank, (idx, prob) in enumerate(zip(top_idxs_adv, top_probs_adv), start=1):
    print(f"{rank}: {imagenet_classes[idx]} (class {idx}) - confidence: {prob:.4f}")

# === PATCH-ONLY L2 AND NNR ===

# 1. L2 on normalized values (patch only)
l2 = (modified[0, ch, y, x] - img_tensor[0, ch, y, x]).pow(2).sqrt().item()

# 2. NNR on unnormalized values (patch only)
orig_unnorm = unnormalize(img_tensor.clone().squeeze()).clamp(0, 1)
mod_unnorm = unnormalize(modified.clone().squeeze()).clamp(0, 1)
nnr = (mod_unnorm[ch, y, x] - orig_unnorm[ch, y, x]).pow(2).sqrt().item()

# === CHECK SUCCESS ===
if adv_top1_idx != true_top1_idx:
    print(f"\n✅ Attack succeeded: top-1 changed from {true_top1_label} to {adv_top1_label}")
else:
    print(f"\n❌ Attack failed: top-1 is still {true_top1_label}")

print(f"\nL2 (patched, normalized): {l2:.6f}")
print(f"NNR (patched, unnormalized): {nnr:.6f}")
