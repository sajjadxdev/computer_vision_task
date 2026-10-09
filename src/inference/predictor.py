import time
import io
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import build_baseline_model, build_transfer_model
from src.data.transforms import get_data_transforms
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("InferencePredictor")


class FireSmokePredictor:
    """
    Production-ready inference pipeline for Fire and Smoke Computer Vision model.
    Handles image loading, preprocessing, model execution, and formatted prediction output.
    """
    def __init__(self, model_type: str = "transfer_learning", checkpoint_path: str = None, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.config["data"]["classes"]
        self.img_size = tuple(self.config["data"]["image_size"])

        # Transforms for inference
        _, self.transform = get_data_transforms(img_size=self.img_size)

        # Select model architecture
        self.model_type = model_type
        if model_type == "baseline_cnn":
            self.model = build_baseline_model(self.config)
            ckpt = checkpoint_path or self.config["models"]["baseline_cnn"]["checkpoint_path"]
        else:
            self.model = build_transfer_model(self.config)
            ckpt = checkpoint_path or self.config["models"]["transfer_learning"]["checkpoint_path"]

        # Load checkpoint if exists
        ckpt_path = Path(ckpt)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded trained model checkpoint from {ckpt_path}")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path}. Using un-trained initialized model.")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_input: Union[bytes, Image.Image, str]) -> Dict[str, Any]:
        """
        Executes inference on an image input (bytes, PIL Image, or filepath).
        Returns predicted label, confidence, probabilities for all classes, and latency.
        """
        start_time = time.time()

        # Convert input to PIL Image
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")

        # Preprocess
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx = int(torch.argmax(outputs, dim=1).item())
        confidence = float(probabilities[pred_idx])
        latency_ms = (time.time() - start_time) * 1000

        prob_dict = {
            self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))
        }

        # Warning trigger if low confidence
        is_uncertain = confidence < self.config["inference"].get("confidence_threshold", 0.50)

        return {
            "prediction": self.classes[pred_idx],
            "confidence": round(confidence, 4),
            "class_probabilities": prob_dict,
            "latency_ms": round(latency_ms, 2),
            "model_type": self.model_type,
            "uncertain_prediction": is_uncertain,
            "device": str(self.device)
        }
