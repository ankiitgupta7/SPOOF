import os
import json
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models as tv_models
from PIL import Image, UnidentifiedImageError
import timm
from tqdm import tqdm
import urllib.request
import pandas as pd

# === ARGPARSE ===
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['ResNet-50', 'VGG16', 'ViT-B_16'],
                    help='Model name to use.')
parser.add_argument('--seed', type=int, default=43,
                    help='Random seed for reproducibility.')
parser.add_argument('--sample', type=int, required=True,
                    help='Sample index (0-based) to process from image folder.')
args = parser.parse_args()

# === CONFIG ===
IMAGE_FOLDER = 'sampled_1000_images_from5000'
MAX_ITER = 10000
DATA_DIR = 'data_collected'

torch.manual_seed(args.seed)

MODEL_CONFIGS = {
    'ResNet-50': lambda: timm.create_model('resnet50', pretrained=True),
    'VGG16': lambda: tv_models.vgg16(pretrained=True),
    'ViT-B_16': lambda: timm.create_model('vit_base_patch16_224', pretrained=True)
}

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



# === ATTACK FUNCTION ===
def one_pixel_attack_until_flipped(img_tensor, target_idx, model, max_iter):
    img = img_tensor.clone().detach()
    _, c, h, w = img.shape
    log_rows = []
    success_gen = None
    adv_at_success = None
    start_time = time.time()

    pbar = tqdm(range(max_iter), desc=" Generations", leave=False, dynamic_ncols=True)
    for i in pbar:
        candidate = img.clone()
        x, y = torch.randint(0, w, (1,)).item(), torch.randint(0, h, (1,)).item()
        ch = torch.randint(0, 3, (1,)).item()
        candidate[0, ch, y, x] = torch.rand(1).item()

        with torch.no_grad():
            probs = F.softmax(model(candidate), dim=1)
            top_probs, top_idxs = probs.topk(5)
            top_probs = top_probs.squeeze(0).tolist()
            top_idxs = top_idxs.squeeze(0).tolist()

        l2 = torch.norm((candidate - img_tensor)).item()
        nnr = torch.norm((candidate - img_tensor).view(-1)).item()
        flipped = top_idxs[0] != target_idx

        if flipped and success_gen is None:
            success_gen = i
            adv_at_success = candidate.clone()

        log_rows.append({
            "iter": i,
            "true_label_idx": target_idx,
            "true_label_name": imagenet_classes[target_idx],
            "pred_1_idx": top_idxs[0],
            "pred_1_name": imagenet_classes[top_idxs[0]],
            "pred_1_conf": top_probs[0],
            "pred_2_idx": top_idxs[1],
            "pred_2_name": imagenet_classes[top_idxs[1]],
            "pred_2_conf": top_probs[1],
            "pred_3_idx": top_idxs[2],
            "pred_3_name": imagenet_classes[top_idxs[2]],
            "pred_3_conf": top_probs[2],
            "pred_4_idx": top_idxs[3],
            "pred_4_name": imagenet_classes[top_idxs[3]],
            "pred_4_conf": top_probs[3],
            "pred_5_idx": top_idxs[4],
            "pred_5_name": imagenet_classes[top_idxs[4]],
            "pred_5_conf": top_probs[4],
            "l2": l2,
            "nnr": nnr,
            "flipped": flipped
        })

        orig_conf = F.softmax(model(img), dim=1)[0, target_idx]
        new_conf = probs[0, target_idx]
        if new_conf < orig_conf:
            img = candidate

    time_taken = time.time() - start_time
    final_probs, final_idxs = F.softmax(model(img), dim=1).topk(5)
    return {
        "final_tensor": img,
        "final_probs": final_probs.squeeze(0).tolist(),
        "final_idxs": final_idxs.squeeze(0).tolist(),
        "log": log_rows,
        "success": success_gen is not None,
        "success_gen": success_gen,
        "adv_at_success": adv_at_success,
        "time_taken": time_taken
    }

# === UTILS ===
def save_images(original, adv, out_dir, tag):
    to_pil = T.ToPILImage()
    unnormalize = T.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    img = lambda x: to_pil(unnormalize(x.squeeze().cpu()))
    img(original).save(os.path.join(out_dir, f"{tag}_original.png"))
    img(adv).save(os.path.join(out_dir, f"{tag}_final.png"))

# === LOAD SELECTED MODEL ===
print(f"Loading model: {args.model}")
model = MODEL_CONFIGS[args.model]().to(device)
model.eval()

# === SELECT IMAGE BY INDEX ===
image_paths = sorted([os.path.join(IMAGE_FOLDER, f)
                      for f in os.listdir(IMAGE_FOLDER)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if args.sample < 0 or args.sample >= len(image_paths):
    raise ValueError(f"Sample index {args.sample} out of range. Total samples: {len(image_paths)}")

img_path = image_paths[args.sample]
image_id = os.path.splitext(os.path.basename(img_path))[0]

# === MAIN PROCESSING ===
try:
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
except (UnidentifiedImageError, OSError):
    raise RuntimeError(f"Unable to load image: {img_path}")

try:
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_idxs = probs.topk(5)
        top_probs = top_probs.squeeze(0).tolist()
        top_idxs = top_idxs.squeeze(0).tolist()

    target_idx = top_idxs[0]
    target_label = imagenet_classes[target_idx]

    result = one_pixel_attack_until_flipped(
        img_tensor, target_idx=target_idx, model=model, max_iter=MAX_ITER)

    replicate_dir = os.path.join(DATA_DIR, args.model,
                             f"sample{args.sample}_{image_id}", f"replicate{args.seed}")

    os.makedirs(replicate_dir, exist_ok=True)
    images_dir = os.path.join(replicate_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    pd.DataFrame(result["log"]).to_csv(os.path.join(replicate_dir, "log.csv"), index=False)

    success_metrics = None
    if result["success_gen"] is not None:
        success_row = result["log"][result["success_gen"]]
        success_metrics = {
            "conf": success_row["pred_1_conf"],
            "l2": success_row["l2"],
            "nnr": success_row["nnr"],
            "predicted_label": success_row["pred_1_name"]
        }

    final_row = result["log"][-1]
    summary = {
        "model": args.model,
        "sample_idx": args.sample,
        "image_file": img_path,
        "true_label": {
            "name": target_label,
            "index": target_idx
        },
        "attack_success": result["success"],
        "success_gen": result["success_gen"],
        "success_metrics": success_metrics,
        "final_metrics": {
            "conf": final_row["pred_1_conf"],
            "l2": final_row["l2"],
            "nnr": final_row["nnr"],
            "predicted_label": final_row["pred_1_name"]
        },
        "time_taken": result["time_taken"]
    }

    with open(os.path.join(replicate_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    save_images(img_tensor, result["final_tensor"], images_dir, tag="adv")

except Exception as e:
    print(f"Error processing image {img_path} with {args.model}: {e}")

print("\n✅ Attack completed.")
