from torchvision import transforms
from typing import Tuple, Dict, Any


def get_data_transforms(img_size: Tuple[int, int] = (224, 224), augment_config: Dict[str, Any] = None):
    """
    Constructs PyTorch Torchvision Image Transformation pipelines for training and evaluation.
    
    Training pipeline incorporates domain-specific augmentations:
    - Random Horizontal Flip
    - Random Rotation
    - Color Jitter (Brightness, Contrast, Saturation)
    - Image Normalization (ImageNet standards)
    """
    if augment_config is None:
        augment_config = {
            "random_horizontal_flip_prob": 0.5,
            "random_rotation_degrees": 15,
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}
        }

    # ImageNet normalization parameters
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=augment_config.get("random_horizontal_flip_prob", 0.5)),
        transforms.RandomRotation(degrees=augment_config.get("random_rotation_degrees", 15)),
        transforms.ColorJitter(
            brightness=augment_config["color_jitter"].get("brightness", 0.2),
            contrast=augment_config["color_jitter"].get("contrast", 0.2),
            saturation=augment_config["color_jitter"].get("saturation", 0.2),
            hue=augment_config["color_jitter"].get("hue", 0.05)
        ),
        transforms.ToTensor(),
        normalize
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, eval_transform
