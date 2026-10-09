import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List, Optional
from src.data.transforms import get_data_transforms
from src.utils.logger import setup_logger

logger = setup_logger("DatasetLoader")


class FireSmokeDataset(Dataset):
    """
    Custom PyTorch Dataset for Fire and Smoke image classification.
    """
    def __init__(self, json_file: str, transform=None):
        with open(json_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.samples[idx]
        image_path = item["path"]
        label = item["label"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            # Fallback black image in case of filesystem issues
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test PyTorch DataLoaders."""
    processed_dir = Path(config["data"]["processed_dir"])
    img_size = tuple(config["data"]["image_size"])
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_transform, eval_transform = get_data_transforms(
        img_size=img_size,
        augment_config=config.get("augmentation", {})
    )

    train_dataset = FireSmokeDataset(processed_dir / "train.json", transform=train_transform)
    val_dataset = FireSmokeDataset(processed_dir / "val.json", transform=eval_transform)
    test_dataset = FireSmokeDataset(processed_dir / "test.json", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    logger.info(f"DataLoaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
