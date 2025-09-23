# ui/gradio_app/infer_mil.py
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# -------------------------
# Utilities
# -------------------------

def _remap_state_dict_keys(sd: dict) -> dict:
    """
    Remap checkpoint keys produced with different naming conventions:
    - 'encoder.*'  -> 'enc.*'
    - 'mil.attention_V' -> 'mil.V', same for U and w
    Keeps everything else unchanged.
    """
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("encoder."):
            nk = "enc." + nk[len("encoder."):]
        # attention heads to short names
        nk = nk.replace("mil.attention_V.", "mil.V.")
        nk = nk.replace("mil.attention_U.", "mil.U.")
        nk = nk.replace("mil.attention_w.", "mil.w.")
        out[nk] = v
    return out


def _list_images(folder_or_file: Path) -> List[Path]:
    p = Path(folder_or_file)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    if p.is_file():
        return [p] if p.suffix.lower() in exts else []

    if p.is_dir():
        return sorted([q for q in p.iterdir() if q.suffix.lower() in exts])

    return []


# -------------------------
# Model definition (MIL with EfficientNet encoder + gated attention)
# -------------------------

class MILAttention(nn.Module):
    """Gated attention pooling (Ilse et al.)."""
    def __init__(self, in_dim: int, hid_dim: int = 256):
        super().__init__()
        self.U = nn.Linear(in_dim, hid_dim)
        self.V = nn.Linear(in_dim, hid_dim)
        self.w = nn.Linear(hid_dim, 1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: [N, D]
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))  # [N, H]
        att = self.w(A).squeeze(-1)                           # [N]
        alpha = torch.softmax(att, dim=0)                     # [N]
        M = torch.sum(alpha.unsqueeze(-1) * H, dim=0)        # [D]
        return M, alpha


class MILNet(nn.Module):
    def __init__(self, backbone: str = "tf_efficientnet_b0_ns", pretrained: bool = False):
        super().__init__()
        # feature extractor
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        feat_dim = self.enc.num_features
        # global pooling after encoder features
        self.gap = nn.AdaptiveAvgPool2d(1)
        # attention pooling over instances and classifier
        self.mil = MILAttention(feat_dim, hid_dim=256)
        self.head = nn.Linear(feat_dim, 1)

    @torch.no_grad()
    def forward_instances(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> features per instance [B, D]
        feats = self.enc(x)                        # [B, D, h, w] or [B, D]
        if feats.ndim == 4:
            feats = self.gap(feats).squeeze(-1).squeeze(-1)  # [B, D]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,3,H,W] instances (one bag)
        H = self.forward_instances(x)    # [N, D]
        M, _ = self.mil(H)               # [D]
        logit = self.head(M).squeeze(-1) # []
        return logit


# -------------------------
# Inference wrapper
# -------------------------

class MILEnsemble:
    """
    Loads fold checkpoints from a directory (mil_v1_fold{0..4}.pt) and
    performs MIL inference on a directory of preprocessed tiles/images.
    Returns (probability, debug_text).
    """
    def __init__(self, ckpt_dir: Path, device: str = "cpu", backbone: str = "tf_efficientnet_b0_ns"):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.backbone = backbone

        self.num_folds = 0
        self.models: List[nn.Module] = []
        self._tfm = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
        ])

        self._load_folds()

    # ---------- public API used by app.py ----------
    def predict(self, processed_dir: Path) -> Tuple[Optional[float], str]:
        """
        processed_dir: directory created by run_pipeline.py that contains tiles or a single preprocessed image.
        Returns (probability_of_indirect, debug_text).
        """
        if not self.models:
            return None, "MIL not loaded"

        imgs = _list_images(processed_dir)
        if not imgs:
            return None, f"MIL: no images under {processed_dir}"

        # Build a single bag from all images in the directory
        bag = []
        for p in imgs:
            try:
                im = Image.open(p).convert("RGB")
                bag.append(self._tfm(im))
            except Exception:
                continue

        if not bag:
            return None, f"MIL: failed to load images in {processed_dir}"

        x = torch.stack(bag, dim=0).to(self.device)  # [N,3,H,W]

        with torch.no_grad():
            logits = []
            for m in self.models:
                m.eval()
                logit = m(x)     # scalar
                logits.append(logit.float().item())
            logit_mean = float(sum(logits) / len(logits))
            prob = float(torch.sigmoid(torch.tensor(logit_mean)).item())

        dbg = f"Instances={x.shape[0]} | folds={len(self.models)} | logits={[round(l,4) for l in logits]}"
        return prob, dbg

    # ---------- internals ----------
    def _load_folds(self):
        if not self.ckpt_dir.exists():
            return

        for k in range(10):  # look for up to 10 folds (usually 5)
            cand = self.ckpt_dir / f"mil_v1_fold{k}.pt"
            if not cand.exists():
                continue
            try:
                ckpt = torch.load(cand, map_location="cpu")
                sd = ckpt.get("state_dict", ckpt)

                # Remap keys to current module names
                sd = _remap_state_dict_keys(sd)

                # Build model and load weights (strict=False to tolerate tiny diffs)
                model = MILNet(self.backbone, pretrained=False)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"[MIL] non-strict load for {cand.name} "
                          f"(missing={len(missing)}, unexpected={len(unexpected)})")

                model.to(self.device)
                self.models.append(model)
                self.num_folds += 1
            except Exception as e:
                print(f"[MIL] strict load failed for {cand.name}, using strict=False ({e})")

        if self.num_folds == 0:
            print("[MIL] no folds found")

