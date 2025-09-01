from pathlib import Path
from src.preprocessing.pipeline import ImagePreprocessor

def test_sample(tmp_path: Path):
    sample_dir = Path("data/raw/images")
    sample_img = next(sample_dir.glob("*.jpg"))
    model_path = Path("models/segmenter/mask_rcnn_molar.pt")
    pre = ImagePreprocessor(model_path)
    info = pre.process_file(sample_img)
    assert info["status"] == "ok"
    assert (tmp_path / info["out_file"]).with_suffix(".jpg")
