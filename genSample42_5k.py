import os
import json
import random
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
OUTPUT_JSON = "imagenet_val_subset_1000_new.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
SUBSET_SIZE = 5000
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

# ----- MAIN PIPELINE -----
def main():
    print(">>> Optimized subset-based evaluation started")

    idx_to_synset, synset_to_name = download_class_index()

    models = []
    for name, constructor in MODEL_CONFIGS:
        print(f"Loading model: {name}")
        model = constructor().eval().to(DEVICE)
        models.append((name, model))

    all_image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".JPEG"))
    assert len(all_image_files) == 50000, "Expected 50K validation images."

    random.seed(SEED)
    subset_indices = random.sample(range(50000), SUBSET_SIZE)
    subset_files = [(i, all_image_files[i]) for i in subset_indices]
    print(f"Selected {SUBSET_SIZE} images for evaluation.")

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_results = []
    for index, img_file in tqdm(subset_files, desc="Evaluating subset"):
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
        passed = True
        result = {
            "index": index,
            "image": img_path,
            "true_label": true_synset,
            "true_label_name": synset_to_name.get(true_synset, "unknown"),
            "predictions": {},
            "bboxes": bboxes
        }

        for name, model in models:
            topk_ids, topk_confs = get_topk_predictions(model, img_tensor)
            topk_synsets = [idx_to_synset[i] for i in topk_ids]

            pred_correct = topk_synsets[0] == true_synset

            result["predictions"][name] = {
                "top1": {
                    "synset": topk_synsets[0],
                    "confidence": round(topk_confs[0], 4),
                    "prediction_truth": pred_correct
                },
                "top5": [
                    {"synset": s, "confidence": round(c, 4)}
                    for s, c in zip(topk_synsets, topk_confs)
                ]
            }

            if not pred_correct:
                passed = False
                break

        if passed:
            all_results.append(result)

    print(f"Correctly predicted by all models: {len(all_results)}")

    random.seed(SEED)
    final_sample = random.sample(all_results, min(SAMPLE_SIZE, len(all_results)))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_sample, f, indent=2)

    print(f"Saved {len(final_sample)} entries to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()