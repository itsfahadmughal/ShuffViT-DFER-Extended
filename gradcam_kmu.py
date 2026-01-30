from __future__ import print_function
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from KMU import KMU
from models.combmodel1 import CombinedModel

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -----------------------------
# Settings (edit if needed)
# -----------------------------
DATASET_TAG = "efficientvitwcc"   # same as in combinemodelkmu.py
MODEL_TAG   = "Ourmodel"
HEAD_TYPE   = "convgap"           # 'baseline', 'mlp', or 'convgap'
FOLD        = 3                   # which fold to visualize
IMG_SIZE    = 224
NUM_IMAGES  = 5                   # how many test images to visualize

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Transforms (match your test transforms)
# -----------------------------
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

# -----------------------------
# Build model (same as training)
# -----------------------------
num_classes = 6
net = CombinedModel(
    num_classes=num_classes,
    head_type=HEAD_TYPE,
    mlp_h1=512,
    mlp_h2=256,
    convgap_mid=512,
)
net.to(device)
net.eval()

print(f"[info] Using head_type = '{HEAD_TYPE}'")

# -----------------------------
# Load checkpoint for given fold
# -----------------------------
root = Path(f"{DATASET_TAG}_{MODEL_TAG}")
ckpt_path = root / str(FOLD) / "Test_model.t7"
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

print(f"[info] Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

# Normalize to a state_dict
if isinstance(checkpoint, nn.Module):
    state = checkpoint.state_dict()
elif isinstance(checkpoint, dict) and 'net' in checkpoint:
    net_obj = checkpoint['net']
    if isinstance(net_obj, nn.Module):
        state = net_obj.state_dict()
    else:
        state = net_obj
elif isinstance(checkpoint, dict):
    state = checkpoint
else:
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

missing, unexpected = net.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[load_state_dict] missing keys:", missing)
    print("[load_state_dict] unexpected keys:", unexpected)

# -----------------------------
# Data loader (test split)
# -----------------------------
testset = KMU(split='Testing', fold=FOLD, transform=transforms_valid)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=True, num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"[info] Test samples in fold {FOLD}: {len(testset)}")

# -----------------------------
# Choose target layer for GradCAM
# Use last conv block of ShuffleNetV2
# -----------------------------
# shufflenet_v2_x1_0 has: conv1, maxpool, stage2, stage3, stage4, conv5, fc
target_layer = net.feature_extractor_shufflenet.conv5  # good high-level conv features

cam = GradCAM(model=net, target_layers=[target_layer])

# -----------------------------
# Output folder
# -----------------------------
out_dir = Path(f"gradcam_{DATASET_TAG}_{HEAD_TYPE}_fold{FOLD}")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[info] Saving Grad-CAM images to: {out_dir}")

# -----------------------------
# Utility: unnormalize for visualization
# -----------------------------
mean_np = np.array(mean).reshape(1, 1, 3)
std_np  = np.array(std).reshape(1, 1, 3)

def tensor_to_rgb(img_tensor):
    """Convert normalized tensor (C,H,W) to RGB numpy [0,1] for visualization."""
    img = img_tensor.cpu().permute(1, 2, 0).numpy()  # (H,W,C)
    img = img * std_np + mean_np
    img = np.clip(img, 0, 1)
    return img

# -----------------------------
# Generate Grad-CAM visualizations
# -----------------------------
net.eval()
count = 0

for idx, (inputs, targets) in enumerate(testloader):
    if count >= NUM_IMAGES:
        break

    inputs, targets = inputs.to(device), targets.to(device)
    img_rgb = tensor_to_rgb(inputs[0])

    # Use true class as target for CAM
    class_id = int(targets[0].item())
    targets_cam = [ClassifierOutputTarget(class_id)]

    grayscale_cam = cam(input_tensor=inputs, targets=targets_cam)[0]  # (H,W)
    cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

    out_path = out_dir / f"gradcam_img{idx}_class{class_id}.png"
    plt.imsave(out_path, cam_image)
    print(f"[info] Saved: {out_path}")
    count += 1

print("[info] Grad-CAM generation done.")
