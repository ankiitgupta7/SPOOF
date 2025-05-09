import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parameters
IMAGE_PATH = 'sampled_1000_images_from5000/ILSVRC2012_val_00013542.JPEG'
PATCH_SIZE = 40  # 40x40 patch
PATCH_POS = (60, 60)  # top-left corner of the patch
ALPHA = 0.3  # transparency level for modification

# Load and normalize image
img = Image.open(IMAGE_PATH).convert('RGB')
img_np = np.array(img).astype(np.float32) / 255.0  # normalize to [0,1]
original = img_np.copy()

# Create an adversarial-style patch (add slight noise or a tint)
i, j = PATCH_POS
patch = original[i:i+PATCH_SIZE, j:j+PATCH_SIZE].copy()

# Example modification: add a red tint (semi-transparent)
tint = np.array([1.0, 0.0, 0.0])  # red
patch_modified = (1 - ALPHA) * patch + ALPHA * tint
modified = original.copy()
modified[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = patch_modified

# Compute L2 and NNR over patch only
diff = patch_modified - patch
l2 = np.linalg.norm(diff)
nnr = np.sum(np.abs(diff))

print(f"Patch-only L2: {l2:.6f}")
print(f"Patch-only NNR: {nnr:.6f}")

# Plot original vs modified region
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(original[i:i+PATCH_SIZE, j:j+PATCH_SIZE])
axs[0].set_title("Original Patch")
axs[1].imshow(modified[i:i+PATCH_SIZE, j:j+PATCH_SIZE])
axs[1].set_title("Modified Patch")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.savefig("patch_comparison.png")
plt.close()

