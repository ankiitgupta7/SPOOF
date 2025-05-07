import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tv_models
from PIL import Image, UnidentifiedImageError
import timm
from tqdm import tqdm
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
IMAGE_FOLDER = 'sampled_1000_images_from5000'
SAMPLE_SIZE = 1
MAX_ITER = 10000
CSV_LOG = 'attack_log.csv'
OUTPUT_IMG_FOLDER = 'attack_images'
os.makedirs(OUTPUT_IMG_FOLDER, exist_ok=True)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_CONFIGS = [
    ('ResNet-50', lambda: timm.create_model('resnet50', pretrained=True)),
    ('VGG16', lambda: tv_models.vgg16(pretrained=True)),
    ('ViT-B/16', lambda: timm.create_model('vit_base_patch16_224', pretrained=True))
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD IMAGENET CLASS LABELS ===
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# === IMAGE TRANSFORM ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# === ONE-PIXEL ATTACK WITH LOGGING ===
def one_pixel_attack_until_flipped(img_tensor, target_idx, model, max_iter=1000):
    img = img_tensor.clone().detach()
    _, c, h, w = img.shape
    class_conf_log = {}
    for i in range(max_iter):
        candidate = img.clone()
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        ch = random.randint(0, 2)
        candidate[0, ch, y, x] = torch.rand(1).item()

        with torch.no_grad():
            probs = F.softmax(model(candidate), dim=1)
            top_probs, top_idxs = probs.topk(3)
            top_probs = top_probs.squeeze(0)
            top_idxs = top_idxs.squeeze(0)

        for idx, prob in zip(top_idxs.tolist(), top_probs.tolist()):
            if idx not in class_conf_log:
                class_conf_log[idx] = []
            class_conf_log[idx].append((i, prob))

        if top_idxs[0].item() != target_idx:
            return candidate, top_idxs.tolist(), top_probs.tolist(), True, class_conf_log

        orig_conf = F.softmax(model(img), dim=1)[0, target_idx]
        new_conf = probs[0, target_idx]
        if new_conf < orig_conf:
            img = candidate

    final_probs, final_idxs = F.softmax(model(img), dim=1).topk(3)
    return img, final_idxs.squeeze(0).tolist(), final_probs.squeeze(0).tolist(), False, class_conf_log

# === GET TOP-K PREDICTIONS ===
def get_topk(model, img_tensor, k=5):
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        topk_probs, topk_idxs = probs.topk(k)
    return topk_probs.squeeze(0).tolist(), topk_idxs.squeeze(0).tolist()

# === SAVE ORIGINAL & ADVERSARIAL IMAGES ===
def save_images(original, adversarial, save_prefix):
    to_pil = T.ToPILImage()
    unnormalize = T.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )
    orig_img = to_pil(unnormalize(original.squeeze().cpu()))
    adv_img = to_pil(unnormalize(adversarial.squeeze().cpu()))
    orig_img.save(os.path.join(OUTPUT_IMG_FOLDER, f"{save_prefix}_original.png"))
    adv_img.save(os.path.join(OUTPUT_IMG_FOLDER, f"{save_prefix}_adversarial.png"))

# === LOAD MODELS ONCE ===
print("Loading models...")
loaded_models = {}
for model_name, model_fn in MODEL_CONFIGS:
    model = model_fn().to(device)
    model.eval()
    loaded_models[model_name] = model

# === SELECT RANDOM IMAGES ===
image_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_paths)
image_paths = image_paths[:SAMPLE_SIZE]

log_rows = []
outer_pbar = tqdm(total=len(image_paths) * len(loaded_models), desc="Processing", dynamic_ncols=True)

for img_path in image_paths:
    try:
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
    except (UnidentifiedImageError, OSError):
        continue

    for model_name, model in loaded_models.items():
        try:
            orig_probs, orig_idxs = get_topk(model, img_tensor, k=5)
            orig_top1_idx = orig_idxs[0]
            orig_top1_label = imagenet_classes[orig_top1_idx]
            orig_top1_conf = orig_probs[0]

            adv_tensor, adv_idxs, adv_probs, flipped, class_log = one_pixel_attack_until_flipped(
                img_tensor, target_idx=orig_top1_idx, model=model, max_iter=MAX_ITER)

            save_prefix = f"{os.path.splitext(os.path.basename(img_path))[0]}_{model_name.replace('/', '_')}"
            save_images(img_tensor, adv_tensor, save_prefix)

            row = {
                'image_path': img_path,
                'model': model_name,
                'orig_class_idx': orig_top1_idx,
                'orig_class_label': orig_top1_label,
                'orig_confidence': orig_top1_conf,
                'adv_class_idx': adv_idxs[0],
                'adv_class_label': imagenet_classes[adv_idxs[0]],
                'adv_confidence': adv_probs[0],
                'flipped': flipped,
                'adv_top2_idx': adv_idxs[1],
                'adv_top2_label': imagenet_classes[adv_idxs[1]],
                'adv_top2_conf': adv_probs[1],
                'adv_top3_idx': adv_idxs[2],
                'adv_top3_label': imagenet_classes[adv_idxs[2]],
                'adv_top3_conf': adv_probs[2]
            }
            log_rows.append(row)
        except Exception:
            continue
        outer_pbar.update(1)

outer_pbar.close()

# === SAVE LOG TO CSV ===
df = pd.DataFrame(log_rows)
df.to_csv(CSV_LOG, index=False)
print(f"\n✅ Attack log saved to {CSV_LOG}")
