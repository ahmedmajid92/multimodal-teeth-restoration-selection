# ui/gradio_app/infer_mil.py
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import timm
import torchvision.transforms as T

def make_tf(img_size, train=False):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4,1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.ColorJitter(0.15,0.15,0.15,0.05),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size*1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.V = nn.Linear(in_dim, hid)
        self.U = nn.Linear(in_dim, hid)
        self.w = nn.Linear(hid, 1)
    def forward(self, H):  # H: (B,K,D)
        AV = torch.tanh(self.V(H))
        AU = torch.sigmoid(self.U(H))
        A  = self.w(AV * AU).squeeze(-1)
        A  = torch.softmax(A, dim=1)
        M  = torch.einsum('bkd,bk->bd', H, A)
        return M, A

class MILNet(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0_ns', drop=0.2):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool='avg')
        d = self.enc.num_features
        self.mil = AttentionMIL(d, 128)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(d, 1)  # some trainings may have used 'classifier' or 'cls_head'
    def forward(self, x):  # x: (B,K,C,H,W)
        B,K,C,H,W = x.shape
        x = x.view(B*K, C, H, W)
        f = self.enc(x)            # (B*K,D)
        f = f.view(B, K, -1)       # (B,K,D)
        bag, A = self.mil(f)       # (B,D)
        bag = self.drop(bag)
        logit = self.head(bag).squeeze(1)
        return logit, A

def _remap_keys(state):
    """Map common head names to 'head'."""
    new = {}
    for k, v in state.items():
        if k.startswith('classifier.'):
            new['head.' + k.split('.',1)[1]] = v
        elif k.startswith('cls_head.'):
            new['head.' + k.split('.',1)[1]] = v
        elif k.startswith('fc.'):
            new['head.' + k.split('.',1)[1]] = v
        else:
            new[k] = v
    return new

class MILEnsemble:
    def __init__(self, ckpt_dir, device='cpu'):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = device
        # accept many naming patterns
        self.ckpts = sorted(list(self.ckpt_dir.glob("*fold*.pt")))
        self.models = []
        self.args = None
        self._load()

    @property
    def num_folds(self): return len(self.models)

    def _load(self):
        loaded = 0
        for ck in self.ckpts:
            try:
                ckpt = torch.load(ck, map_location='cpu', weights_only=False)
                a = ckpt.get('args', {})
                m = MILNet(backbone=a.get('backbone','tf_efficientnet_b0_ns'), drop=a.get('dropout',0.2)).to(self.device)
                state = ckpt['model']
                # try strict
                try:
                    m.load_state_dict(state, strict=True)
                except RuntimeError:
                    # try remap
                    state2 = _remap_keys(state)
                    try:
                        m.load_state_dict(state2, strict=True)
                    except RuntimeError as e:
                        # last resort
                        print(f"[MIL] strict load failed for {ck.name}, using strict=False ({e})")
                        m.load_state_dict(state2, strict=False)
                m.eval()
                img_size = a.get('img_size', 320)
                K = a.get('instances', 12)
                self.models.append((m, img_size, K))
                loaded += 1
            except Exception as e:
                print(f"[MIL] skip {ck.name}: {e}")
        print(f"[MIL] loaded folds: {loaded}")

    def _build_bag(self, img: Image.Image, img_size: int, K: int):
        tf = make_tf(img_size, train=False)
        crops = [tf(img) for _ in range(K)]
        x = torch.stack(crops, 0)  # (K,C,H,W)
        return x

    def predict(self, image_path: Path):
        if not self.models: 
            return 0.5, "MIL not loaded"
        img = Image.open(image_path).convert('RGB')
        probs = []
        for m, img_size, K in self.models:
            bag = self._build_bag(img, img_size, K).unsqueeze(0).to(self.device)  # (1,K,C,H,W)
            with torch.no_grad():
                plist = []
                for flip in [False, True]:
                    b = bag
                    if flip: b = torch.flip(b, dims=[3])
                    logit, _ = m(b)
                    plist.append(torch.sigmoid(logit).item())
                probs.append(float(np.mean(plist)))
        return float(np.mean(probs)), f"fold_probs={np.round(probs,3)}"
