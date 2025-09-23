from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import timm

TAB_FEATURES = [
    'depth','width','enamel_cracks','occlusal_load','carious_lesion',
    'opposing_type','adjacent_teeth','age_range','cervical_lesion'
]

def timm_tf(img_size, train=False):
    return timm.data.create_transform(
        input_size=img_size, is_training=train,
        interpolation='bicubic',
        mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
    )

class MMNet(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b4_ns', tab_in=9, tab_hidden=64, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool='avg')
        d = self.backbone.num_features
        self.tab = nn.Sequential(
            nn.Linear(tab_in, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(True),
        )
        self.fuse = nn.Dropout(drop)
        self.cls_head = nn.Linear(d + tab_hidden, 1)
        self.reg_head = nn.Linear(d + tab_hidden, 1)
    def forward(self, xi, xt):
        fi = self.backbone(xi)
        ft = self.tab(xt)
        f  = self.fuse(torch.cat([fi, ft], 1))
        return self.cls_head(f).squeeze(1), self.reg_head(f).squeeze(1)

class MMEnsemble:
    def __init__(self, ckpt_dir, device='cpu'):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = device
        self.ckpts = sorted(self.ckpt_dir.glob("mm_dualtask_fold*.pt"))
        self.models = []
        self.scales = []  # (mean, scale)
        self.img_size = None
        self.batch_size = None
        self._load()

    @property
    def num_folds(self): return len(self.models)

    def _load(self):
        for ck in self.ckpts:
            ckpt = torch.load(ck, map_location='cpu', weights_only=False)
            a = ckpt['args']
            model = MMNet(backbone=a['backbone'], tab_in=len(TAB_FEATURES),
                          tab_hidden=a['tab_hidden'], drop=a['dropout']).to(self.device)
            # match key names from training
            state = ckpt['model']
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append((model, ckpt['T']))  # store temperature

            mean = np.array(ckpt['scaler_mean']) if ckpt.get('scaler_mean') is not None else np.zeros(len(TAB_FEATURES))
            scale = np.array(ckpt['scaler_scale']) if ckpt.get('scaler_scale') is not None else np.ones(len(TAB_FEATURES))
            self.scales.append((mean, scale))
            self.img_size = a['img_size']

        if not self.models:
            print("[MMEnsemble] No checkpoints found.")

    def _prep_tab(self, tab_dict, fold=0):
        mean, scale = self.scales[fold]
        if tab_dict is None:
            # Use means so standardized vector ~ zeros (image-only behavior)
            x = mean.copy()
        else:
            x = np.array([float(tab_dict[k]) for k in TAB_FEATURES], dtype=np.float32)
        z = (x - mean) / np.where(scale == 0, 1.0, scale)
        return torch.tensor(z, dtype=torch.float32).unsqueeze(0)

    def predict(self, image_path: Path, tab_dict=None):
        """Return (prob_mm, debug_str)."""
        if not self.models:
            return 0.5, "MM not loaded"
        img = Image.open(image_path).convert('RGB')
        tf = timm_tf(self.img_size, train=False)
        xi = tf(img).unsqueeze(0).to(self.device)

        probs = []
        for f, (model, T) in enumerate(self.models):
            xt = self._prep_tab(tab_dict, fold=f).to(self.device)
            with torch.no_grad():
                # TTA: original + h+v flips
                logits = []
                for flip in [None, 'h', 'v']:
                    x = xi.clone()
                    if flip == 'h': x = torch.flip(x, dims=[3])
                    elif flip == 'v': x = torch.flip(x, dims=[2])
                    logit, _ = model(x, xt)
                    logits.append(logit)
                logit = torch.stack(logits, 0).mean(0)  # (1,)
                p = torch.sigmoid(logit / T).item()
                probs.append(p)
        prob = float(np.mean(probs))
        return prob, f"fold_probs={np.round(probs,3)}"
