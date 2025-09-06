# models/vision/model_factory.py
import timm

def create_model(model_name: str = "tf_efficientnet_b3_ns",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 drop_rate: float = 0.2,
                 drop_path_rate: float = 0.1,
                 in_chans: int = 3):
    """
    Return a timm model whose internal classifier head is correctly set.
    For soft labels, pass num_classes=1 (single logit).
    """
    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,   # <-- IMPORTANT: let timm build the classifier
        in_chans=in_chans,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    return m
