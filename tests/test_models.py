import pytest
import torch
from src.utils.config import load_config
from src.models import build_baseline_model, build_transfer_model


def test_baseline_cnn_forward_pass():
    config = load_config("config/config.yaml")
    model = build_baseline_model(config)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2)
    assert not torch.isnan(output).any()


def test_transfer_model_forward_pass():
    config = load_config("config/config.yaml")
    model = build_transfer_model(config)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 2)
    assert not torch.isnan(output).any()
