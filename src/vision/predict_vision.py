# src/vision/predict_vision.py
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

ARCH_MAP_BY_STEM = {
    32: ["tf_efficientnet_b0_ns","tf_efficientnet_b1_ns","tf_efficientnet_b2_ns"],
    40: ["tf_efficientnet_b3_ns"],
    48: ["tf_efficientnet_b4_ns"],
    56: ["tf_efficientnet_b5_ns"],
}

def _strip_prefix(k: str) -> str:
    for pref in ("model.", "module."):
        if k.startswith(pref):
            return k[len(pref):]
    return k

def _infer_arch_and_outputs(sd: dict):
    # sd is a plain state_dict (string -> tensor)
    stem_keys = [k for k in sd.keys() if k.endswith("conv_stem.weight")]
    num_out = None
    # try to detect classifier outputs
    for headk in ("classifier.weight","fc.weight","head.fc.weight","head.weight"):
        if headk in sd:
            num_out = sd[headk].shape[0]
            break
    archs = None
    if stem_keys:
        stem_out = sd[stem_keys[0]].shape[0]
        archs = ARCH_MAP_BY_STEM.get(stem_out)
    # sensible fallbacks
    return (archs or ["tf_efficientnet_b3_ns","tf_efficientnet_b0_ns"]), (num_out or 1)

def _try_build_model(arch_candidates, num_outputs):
    last_err = None
    for arch in arch_candidates:
        try:
            m = timm.create_model(arch, pretrained=False, num_classes=num_outputs)
            return m, arch
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to instantiate any arch from {arch_candidates}: {last_err}")

def _load_state_dict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # TorchScript?
    if isinstance(ckpt, torch.jit.ScriptModule):
        return None, ckpt.eval().to(device), True  # (sd, model, is_script)
    # plain or lightning
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else:
        sd = ckpt
    sd = { _strip_prefix(k): v for k, v in sd.items() }
    return sd, None, False

def _tf(size=224):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

@torch.inference_mode()
def predict_image(ckpt_path, image_path, is_regressor=False, size=224):
    ckpt_path = str(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sd, script_model, is_script = _load_state_dict(ckpt_path, device)
    if is_script:
        model = script_model
        arch = "script"
        num_outputs = None
    else:
        arch_candidates, num_outputs = _infer_arch_and_outputs(sd)
        model, arch = _try_build_model(arch_candidates, num_outputs)
        # allow partial matches
        model.load_state_dict(sd, strict=False)
        model.eval().to(device)

    img = Image.open(image_path).convert("RGB")
    x = _tf(size)(img).unsqueeze(0).to(device)
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    y = y.squeeze()

    if is_regressor:
        # treat as probability-like regression in [0,1]
        return float(torch.clamp(y, 0, 1).detach().cpu().item())

    # classifier → probability of "Indirect"
    if y.dim() == 0:
        prob = torch.sigmoid(y)
    elif y.numel() == 2:
        prob = F.softmax(y, dim=0)[1]
    else:
        prob = F.softmax(y, dim=0)[-1]
    return float(prob.detach().cpu().item())
