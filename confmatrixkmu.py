from __future__ import print_function

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix, classification_report

from KMU import KMU
from models.combmodel1 import CombinedModel


# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description='KMU-FED Confusion Matrix (ShuffViT-DFER)')
parser.add_argument('--dataset', type=str, default='shufflnetw1',
                    help='dataset tag used in checkpoint folder name (e.g., efficientvitwcc)')
parser.add_argument('--model', type=str, default='Ourmodel',
                    choices=['Ourmodel', 'efficientViT'],
                    help='model tag used in checkpoint folder name')
parser.add_argument('--ckpt_root', type=str, default=None,
                    help='override checkpoint root (e.g., efficientvitwcc_Ourmodel)')
parser.add_argument('--start_fold', type=int, default=1, help='first fold (1-indexed)')
parser.add_argument('--end_fold', type=int, default=10, help='last fold (inclusive)')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=int, default=224)

# 🔹 NEW: head selection + hyperparams (must match training)
parser.add_argument('--head', type=str, default='mlp',
                    choices=['baseline', 'mlp', 'convgap'],
                    help='Classifier head type for evaluation')
parser.add_argument('--mlp_h1', type=int, default=512)
parser.add_argument('--mlp_h2', type=int, default=256)
parser.add_argument('--convgap_mid', type=int, default=512)

args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Transforms
# -----------------------------
transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((args.img_size, args.img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
])


# -----------------------------
# Plot util
# -----------------------------
def plot_confusion_matrix(cm, classes, normalize=True,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()


# -----------------------------
# Model
# -----------------------------
num_classes = 6
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

if args.model == 'Ourmodel':
    net = CombinedModel(
        num_classes=num_classes,
        head_type=args.head,
        mlp_h1=args.mlp_h1,
        mlp_h2=args.mlp_h2,
        convgap_mid=args.convgap_mid,
    )
else:
    # If you ever use a plain EfficientViT backbone:
    import timm
    net = timm.create_model('efficientvit_m4.r224_in1k', pretrained=True, num_classes=num_classes)

net.to(device)
net.eval()

# -----------------------------
# Paths
# -----------------------------
if args.ckpt_root is not None:
    root = Path(args.ckpt_root)
else:
    # Note: head is NOT in folder name because your training uses dataset_model only
    root = Path(f'{args.dataset}_{args.model}')

print(f'[info] checkpoint root = {root.resolve()}')
os.makedirs(root, exist_ok=True)

# -----------------------------
# Eval across folds
# -----------------------------
all_predicted = []
all_targets = []
found_any = False

for fold in range(args.start_fold, args.end_fold + 1):
    fold_dir = root / f'{fold}'
    ckpt_path = fold_dir / 'Test_model.t7'
    if not ckpt_path.exists():
        print(f'[warn] missing checkpoint: {ckpt_path} -> skipping fold {fold}')
        continue

    print(f'[info] Fold {fold}: loading {ckpt_path}')

    # --- load checkpoint robustly (works with old and new format) ---
    # You created these checkpoints, so weights_only=False is fine.
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
        raise TypeError(f'Unsupported checkpoint type: {type(checkpoint)}')

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('[load_state_dict] missing keys:', missing)
        print('[load_state_dict] unexpected keys:', unexpected)

    # --- data for this fold ---
    trainset = KMU(split='Training', fold=fold, transform=transforms_valid)
    testset  = KMU(split='Testing',  fold=fold, transform=transforms_valid)
    print(len(trainset), len(testset))
    print(f'Fold {fold}: Train samples: {len(trainset)}, Test samples: {len(testset)}')

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --- inference ---
    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predicted.append(predicted.detach().cpu())
            all_targets.append(targets.detach().cpu())

    found_any = True

if not found_any:
    raise FileNotFoundError(f'No checkpoints found under {root}/<fold>/Test_model.t7')

# -----------------------------
# Metrics + Confusion Matrix
# -----------------------------
all_predicted = torch.cat(all_predicted, dim=0).numpy()
all_targets   = torch.cat(all_targets,   dim=0).numpy()

acc = 100.0 * (all_predicted == all_targets).mean()
print(f'Accuracy: {acc:.3f}%')

print('Classification Report:\n',
      classification_report(all_targets, all_predicted, target_names=class_names))

cm = confusion_matrix(all_targets, all_predicted)

plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title=f'Confusion Matrix (Accuracy: {acc:.3f}%)')
out_png = root / 'confusion_matrix_kmu.png'
plt.savefig(out_png, bbox_inches='tight', dpi=200)
plt.close()
print(f'[info] saved: {out_png}')
