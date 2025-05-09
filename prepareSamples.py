import os
import json
import random
import shutil
import csv
import torch
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import timm
import torchvision.models as tv_models
from torch.nn import functional as F
from tqdm import tqdm
from urllib.request import urlretrieve

# ----- CONFIG -----
IMAGE_DIR = "ILSVRC2012_img_val"
ANNOTATION_DIR = os.path.join("ILSVRC2012_bbox_val_v3", "val")
OUTPUT_JSON_1000 = "imagenet_val_all_correct_1000.json"
OUTPUT_JSON_50K = "imagenet_val_all_results_50k.json"
OUTPUT_CSV_STATS = "imagenet_model_stats.csv"
SAMPLED_IMAGE_DIR = "sampled_1000_images_from5000"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
SAMPLE_SIZE = 1000
SEED = 42

MODEL_CONFIGS = [
    ("ResNet-50", lambda: timm.create_model("resnet50", pretrained=True)),
    ("VGG16", lambda: tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)),
    ("ViT-B/16", lambda: timm.create_model("vit_base_patch16_224", pretrained=True))
]

# ----- UTILITIES -----
def download_class_index(path="imagenet_class_index.json"):
    if not os.path.exists(path):
        print("Downloading ImageNet class index mapping...")
        urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", path)
    with open(path) as f:
        class_index = json.load(f)
    idx_to_synset = {int(k): v[0] for k, v in class_index.items()}
    synset_to_name = {v[0]: v[1] for k, v in class_index.items()}
    return idx_to_synset, synset_to_name

def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    synsets = [obj.find("name").text for obj in root.findall("object")]
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        boxes.append({
            "xmin": int(bbox.find("xmin").text),
            "ymin": int(bbox.find("ymin").text),
            "xmax": int(bbox.find("xmax").text),
            "ymax": int(bbox.find("ymax").text)
        })
    return synsets[0] if synsets else None, boxes

def get_topk_predictions(model, img_tensor, k=5):
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(DEVICE))
        probs = F.softmax(logits, dim=1)
        confs, indices = probs.topk(k)
        return indices[0].tolist(), confs[0].tolist()

def copy_sampled_images(sample_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(sample_json, "r") as f:
        data = json.load(f)

    copied = 0
    for entry in tqdm(data, desc="Copying sampled images"):
        src = entry["image"]
        dst = os.path.join(output_dir, os.path.basename(src))
        try:
            shutil.copy(src, dst)
            copied += 1
        except Exception as e:
            print(f"Failed to copy {src}: {e}")

    print(f"Copied {copied} images to '{output_dir}'")

# ----- MAIN PIPELINE -----
def main():
    print("Starting evaluation...")

    idx_to_synset, synset_to_name = download_class_index()

    models = []
    for name, constructor in MODEL_CONFIGS:
        print(f"Loading model: {name}")
        model = constructor().eval().to(DEVICE)
        models.append((name, model))

    all_image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".JPEG"))
    assert len(all_image_files) == 50000, "Expected 50K validation images."

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    correct_counts = {name: 0 for name, _ in models}
    total_confidences = {name: 0.0 for name, _ in models}
    total_images = 0
    all_results = []
    jointly_correct_results = []

    for img_file in tqdm(all_image_files, desc="Evaluating images"):
        xml_file = img_file.replace(".JPEG", ".xml")
        xml_path = os.path.join(ANNOTATION_DIR, xml_file)
        img_path = os.path.join(IMAGE_DIR, img_file)

        if not os.path.exists(xml_path):
            continue

        true_synset, bboxes = parse_xml_annotation(xml_path)
        if not true_synset:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_tensor = transform(image)
        total_images += 1
        passed_all = True
        result = {
            "image": img_path,
            "true_label": true_synset,
            "true_label_name": synset_to_name.get(true_synset, "unknown"),
            "predictions": {},
            "bboxes": bboxes
        }

        for name, model in models:
            topk_ids, topk_confs = get_topk_predictions(model, img_tensor)
            topk_synsets = [idx_to_synset[i] for i in topk_ids]
            is_correct = topk_synsets[0] == true_synset
            if is_correct:
                correct_counts[name] += 1
            else:
                passed_all = False
            total_confidences[name] += topk_confs[0]

            result["predictions"][name] = {
                "top1": {
                    "synset": topk_synsets[0],
                    "confidence": round(topk_confs[0], 4),
                    "prediction_truth": is_correct
                },
                "top5": [
                    {"synset": s, "confidence": round(c, 4)}
                    for s, c in zip(topk_synsets, topk_confs)
                ]
            }

        all_results.append(result)
        if passed_all:
            jointly_correct_results.append(result)

    print(f"Jointly correct predictions: {len(jointly_correct_results)} / {total_images}")

    # Save full 50k results
    with open(OUTPUT_JSON_50K, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save 1000-sample subset
    random.seed(SEED)
    final_sample = random.sample(jointly_correct_results, min(SAMPLE_SIZE, len(jointly_correct_results)))
    with open(OUTPUT_JSON_1000, "w") as f:
        json.dump(final_sample, f, indent=2)

    # Compute statistics for both datasets
    sample_counts = {name: 0 for name, _ in models}
    sample_confs = {name: 0.0 for name, _ in models}
    for entry in final_sample:
        for name, _ in models:
            pred_info = entry["predictions"][name]["top1"]
            if pred_info["prediction_truth"]:
                sample_counts[name] += 1
            sample_confs[name] += pred_info["confidence"]

    # Save CSV summary
    with open(OUTPUT_CSV_STATS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Top-1 Accuracy (%)", "Avg. Confidence"])
        for name in correct_counts:
            writer.writerow([
                name, "Full-50k",
                f"{correct_counts[name] / total_images * 100:.2f}",
                f"{total_confidences[name] / total_images:.4f}"
            ])
            writer.writerow([
                name, "Sample-1000",
                f"{sample_counts[name] / len(final_sample) * 100:.2f}",
                f"{sample_confs[name] / len(final_sample):.4f}"
            ])

    # Copy sampled images to target folder
    copy_sampled_images(OUTPUT_JSON_1000, SAMPLED_IMAGE_DIR)

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
