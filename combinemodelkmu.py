from __future__ import print_function
import os, time, ssl, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Project imports
from KMU import KMU
from models.combmodel1 import CombinedModel  # uses head_type, mlp_h1, mlp_h2, convgap_mid
import utils

# Optional: silence SSL for model weights download (your original line)
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description='PyTorch KMUFED ShuffViT-DFER Training')
parser.add_argument('--model', type=str, default='Ourmodel', help='(kept for folder naming)')
parser.add_argument('--dataset', type=str, default='efficientvitwcc', help='dataset tag for folder naming')
parser.add_argument('--fold', default=1, type=int, help='k fold number (1-indexed)')
parser.add_argument('--bs', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# NEW: head selection + tunables
parser.add_argument('--head', type=str, default='mlp',
                    choices=['baseline', 'mlp', 'convgap'],
                    help='Classifier head after fusion')
parser.add_argument('--mlp_h1', type=int, default=512)
parser.add_argument('--mlp_h2', type=int, default=256)
parser.add_argument('--convgap_mid', type=int, default=512)

# (optional) image size
parser.add_argument('--img_size', type=int, default=224)

opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"

best_Test_acc = 0.0
best_Test_acc_epoch = 0
start_epoch = 0

train_accuracy_values, test_accuracy_values = [], []
train_loss_values, test_loss_values = [], []

total_epoch = 25  # was 90

# Save dir (matches your existing structure)
path_root = f"{opt.dataset}_{opt.model}"
path = os.path.join(path_root, str(opt.fold))

# -----------------------------
# Data
# -----------------------------
print('==> Preparing data..')
print('CUDA available:', use_cuda)

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((opt.img_size, opt.img_size)),   # FIXED: (224,) -> (224, 224)
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# For training data , we add some augmentation
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((opt.img_size, opt.img_size)),   # FIXED
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(40),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    torchvision.transforms.RandomAffine(degrees=40, scale=(.3, 1.1), shear=0.15),
    torchvision.transforms.GaussianBlur(kernel_size=5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = KMU(split='Training', fold=opt.fold, transform=transforms_train)
testset  = KMU(split='Testing',  fold=opt.fold, transform=transforms_valid)

# You can increase workers if your system supports it
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0, pin_memory=use_cuda)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=32,     shuffle=False, num_workers=0, pin_memory=use_cuda)

# -----------------------------
# Model
# -----------------------------
num_classes = 6
net = CombinedModel(
    num_classes=num_classes,
    head_type=opt.head,          # NEW
    mlp_h1=opt.mlp_h1,           # NEW
    mlp_h2=opt.mlp_h2,           # NEW
    convgap_mid=opt.convgap_mid  # NEW
)
net.to(device)

# -----------------------------
# Resume (always prefer pure state_dict)
# -----------------------------
if opt.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), f'Error: no checkpoint directory found at {path}'
    ckpt_path = os.path.join(path, 'Test_model.t7')
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'net' in checkpoint and isinstance(checkpoint['net'], dict):
        state = checkpoint['net']
    elif isinstance(checkpoint, dict):
        # older format may have saved the full model; normalize
        from torch import nn
        state = checkpoint['net'].state_dict() if isinstance(checkpoint.get('net', None), nn.Module) else checkpoint
    else:
        # very old: full model
        state = checkpoint.state_dict()
    net.load_state_dict(state, strict=False)
    best_Test_acc = float(checkpoint.get('best_Test_acc', 0.0))
    best_Test_acc_epoch = int(checkpoint.get('best_Test_acc_epoch', 0))
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

# -----------------------------
# Helpers
# -----------------------------
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    h = int(elapsed_time // 3600)
    m = int((elapsed_time - h * 3600) // 60)
    s = int(elapsed_time - h * 3600 - m * 60)
    return h, m, s

total_processing_time_train = 0.0
total_processing_time_test  = 0.0

# -----------------------------
# Train / Test
# -----------------------------
def train(epoch):
    print('\nEpoch:', epoch)
    global total_processing_time_train
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        t0 = time.time()
        outputs = net(inputs)
        t1 = time.time()
        total_processing_time_train += (t1 - t0)
        loss = criterion(outputs, targets)
        loss.backward()
        # utils.clip_gradient(optimizer, 0.1)   # keep commented unless you need it
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        utils.progress_bar(batch_idx, len(trainloader),
            'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total
    train_accuracy_values.append(Train_acc)
    train_loss_values.append(train_loss/(batch_idx+1))

def test(epoch):
    global best_Test_acc, best_Test_acc_epoch, total_processing_time_test
    net.eval()
    PrivateTest_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            t0 = time.time()
            outputs = net(inputs)
            t1 = time.time()
            total_processing_time_test += (t1 - t0)

            loss = criterion(outputs, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            utils.progress_bar(batch_idx, len(testloader),
                'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
                % (PrivateTest_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Test_acc = 100.*correct/total
    test_accuracy_values.append(Test_acc)
    test_loss_values.append(PrivateTest_loss/(batch_idx+1))

    # --- Save checkpoint: ALWAYS pure state_dict ---
    if Test_acc > best_Test_acc:
        print('Saving..  best_Test_acc: %0.3f' % Test_acc)
        os.makedirs(path, exist_ok=True)
        state = {
            'net': net.state_dict(),
            'best_Test_acc': float(Test_acc),
            'best_Test_acc_epoch': int(epoch),
        }
        torch.save(state, os.path.join(path, 'Test_model.t7'))
        best_Test_acc = float(Test_acc)
        best_Test_acc_epoch = int(epoch)

    # timing diagnostics
    num_training_samples = len(trainloader.dataset)
    num_testing_samples  = len(testloader.dataset)
    avg_train = total_processing_time_train / max(1, num_training_samples)
    avg_test  = total_processing_time_test  / max(1, num_testing_samples)
    # Note: avg per-image includes only forward time here (not data loading etc.)
    print(f'Avg per-image forward time (Train): {avg_train:.6f}s | (Test): {avg_test:.6f}s')

# -----------------------------
# Main loop
# -----------------------------
total_start_time = time.monotonic()
for epoch in range(start_epoch, total_epoch):
    ep_t0 = time.monotonic()
    train(epoch)
    test(epoch)
    ep_t1 = time.monotonic()
    eh, em, es = epoch_time(ep_t0, ep_t1)
    print(f'Epoch: {epoch+1:02} | Time: {eh}h {em}m {es}s')

total_end_time = time.monotonic()
th, tm, ts = epoch_time(total_start_time, total_end_time)
total_time_estimate_hours = th + tm/60 + ts/3600
print(f'Total Time: {th}h {tm}m {ts}s | ~{total_time_estimate_hours:.2f} hours')

print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)
