import pytest
import torch
from pathlib import Path
from src.utils.config import load_config
from src.data.transforms import get_data_transforms
from src.data.download import prepare_dataset
from src.data.dataset import create_dataloaders


def test_data_preparation_and_transforms():
    config = load_config("config/config.yaml")
    summary = prepare_dataset("config/config.yaml")

    assert summary["total_samples"] > 0
    assert "fire" in summary["class_distribution"]
    assert "smoke" in summary["class_distribution"]

    train_transform, eval_transform = get_data_transforms(img_size=(128, 128))
    assert train_transform is not None
    assert eval_transform is not None


def test_dataloaders():
    config = load_config("config/config.yaml")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    for images, labels, paths in train_loader:
        assert images.shape[0] <= config["training"]["batch_size"]
        img_size = tuple(config["data"]["image_size"])
        assert images.shape[1:] == (3,) + img_size
        assert labels.shape[0] == images.shape[0]
        break
