import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from models.build import EfficientViT_M2  # your repo’s EfficientViT-M2

# ---------- Heads (extensions) ----------
class MLPHead(nn.Module):
    """Two-layer MLP head: FC -> BN -> ReLU -> Dropout -> FC -> BN -> ReLU -> Dropout -> FC"""
    def __init__(self, in_dim, num_classes, h1=512, h2=256, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h2, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class ConvGAPHead(nn.Module):
    """1x1 Conv -> BN -> ReLU -> GAP -> FC"""
    def __init__(self, in_dim, num_classes, mid=512):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, mid, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(mid)
        self.act  = nn.ReLU(inplace=True)
        self.fc   = nn.Linear(mid, num_classes)
    def forward(self, x):                 # x: (N, C)
        x = x.unsqueeze(-1).unsqueeze(-1) # -> (N, C, 1, 1)
        x = self.proj(x)
        x = self.bn(x)
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (N, mid)
        return self.fc(x)

class BaselineHead(nn.Module):
    """Single FC baseline head."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# ---------- Combined Model ----------
class CombinedModel(nn.Module):
    """
    ShuffleNetV2 (CNN) + EfficientViT-M2 (ViT) with selectable classifier head after fusion:
      - baseline  : single FC
      - mlp       : two-layer MLP (your current classifier)
      - convgap   : 1x1 Conv -> BN -> ReLU -> GAP -> FC
    """
    def __init__(self, num_classes=6, head_type='mlp', mlp_h1=512, mlp_h2=256, convgap_mid=512, drop_p=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.head_type   = head_type

        # --- ShuffleNetV2 backbone ---
        shuf = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        shuf_out = shuf.fc.in_features  # classifier input dim (usually 1024)
        shuf.fc = nn.Identity()         # we want pooled features
        self.feature_extractor_shufflenet = shuf

        # --- EfficientViT-M2 backbone (repo build) ---
        vit = EfficientViT_M2(pretrained='efficientvit_m2')
        vit_out = vit.head.l.in_features
        vit.head.l = nn.Identity()
        self.feature_extractor_efficientvit = vit

        # --- Fusion ---
        self.fused_dim = shuf_out + vit_out

        # --- Heads ---
        if head_type == 'baseline':
            self.classifier = BaselineHead(self.fused_dim, num_classes)
        elif head_type == 'mlp':
            self.classifier = MLPHead(self.fused_dim, num_classes, h1=mlp_h1, h2=mlp_h2, p=drop_p)
        elif head_type == 'convgap':
            self.classifier = ConvGAPHead(self.fused_dim, num_classes, mid=convgap_mid)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # --- Parameter accounting (informative prints) ---
        shuff_params = sum(p.numel() for p in self.feature_extractor_shufflenet.parameters())
        efficientvit_params = sum(p.numel() for p in self.feature_extractor_efficientvit.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        print("ShuffleNet Parameters:", shuff_params)
        print("EfficientViT Parameters:", efficientvit_params)
        print("Classifier Parameters:", classifier_params)
        print("Total Parameters:", shuff_params + efficientvit_params + classifier_params)
        print(f"[Head] Using '{self.head_type}'")

    def forward(self, x):
        f_shuf = self.feature_extractor_shufflenet(x)   # (N, C1)
        f_vit  = self.feature_extractor_efficientvit(x) # (N, C2)
        fused  = torch.cat((f_shuf, f_vit), dim=1)      # (N, C1+C2)
        logits = self.classifier(fused)
        return logits
