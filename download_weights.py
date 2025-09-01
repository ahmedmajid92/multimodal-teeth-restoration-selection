# download_weights.py
#!/usr/bin/env python3
"""
Download the COCO-pretrained Mask R-CNN via torchvision API
and save its state_dict to models/segmenter/mask_rcnn_molar.pt
"""
import torch
from pathlib import Path
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)

def main():
    dst = Path("models/segmenter/mask_rcnn_molar.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print("⏳ Loading Mask R-CNN (ResNet-50 FPN) pretrained on COCO…")
    # Use the new torchvision weights enum
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()

    print(f"💾 Saving state_dict to {dst}")
    torch.save(model.state_dict(), dst)
    print("✅ Done — you can now run the preprocessing pipeline.\n"
          "   python run_pipeline.py --input_dir data/raw/images \\\n"
          "                         --output_dir data/processed/images \\\n"
          "                         --model_path models/segmenter/mask_rcnn_molar.pt")

if __name__ == "__main__":
    main()
