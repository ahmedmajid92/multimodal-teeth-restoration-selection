# src/vision/predict_vision.py
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

def _default_tf(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

def _load_model_from_ckpt(ckpt_path, num_outputs=1):
    """
    Tries: torch.jit, then torch state_dict with built-in small head (linear->sigmoid).
    Uses a tiny generic CNN head if arch info isn't present.
    """
    ckpt_path = Path(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = torch.jit.load(str(ckpt_path), map_location=device)
        model.eval()
        return model.to(device), device, True
    except Exception:
        pass

    # Fallback: a minimal ConvNet head to accept state_dict keys ('model', 'state_dict', etc.)
    import timm
    arch = "tf_efficientnet_b0_ns"
    model = timm.create_model(arch, pretrained=False, num_classes=num_outputs)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    # allow missing/unexpected keys
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(device), device, False

@torch.inference_mode()
def predict_image(ckpt_path, image_path, is_regressor=False, size=224):
    model, device, _ = _load_model_from_ckpt(ckpt_path, num_outputs=1)
    img = Image.open(image_path).convert("RGB")
    x = _default_tf(size)(img).unsqueeze(0).to(device)
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    y = y.squeeze()
    if not is_regressor:
        y = torch.sigmoid(y)
    return float(y.detach().cpu().item())  # probability for "Indirect"
