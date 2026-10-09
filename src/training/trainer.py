import time
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("Trainer")


class Trainer:
    """
    Computations-aware Training Engine for PyTorch Computer Vision models.
    Supports metrics collection, validation tracking, early stopping, and checkpoint saving.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], model_name: str = "baseline"):
        self.config = config
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else config["training"].get("device", "cpu"))
        self.model = model.to(self.device)

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.learning_rate = train_cfg["learning_rate"]
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)

        # Loss function with slight label smoothing to prevent overconfidence
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Optimizer
        if train_cfg.get("optimizer", "adamw").lower() == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Scheduler
        if train_cfg.get("scheduler") == "step":
            self.scheduler = StepLR(self.optimizer, step_size=train_cfg.get("step_size", 3), gamma=train_cfg.get("gamma", 0.5))
        elif train_cfg.get("scheduler") == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        else:
            self.scheduler = None

        # Early Stopping setup
        self.patience = train_cfg.get("early_stopping_patience", 4)
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels, _ in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels, _ in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        total_loss = running_loss / len(data_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        return {
            "loss": total_loss,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str) -> Dict[str, Any]:
        logger.info(f"Starting training for '{self.model_name}' on device [{self.device}]...")
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

        start_time = time.time()
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_score"])

            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}"
            )

            # Checkpoint Best Model
            if val_metrics["f1_score"] > self.best_val_f1 or val_metrics["loss"] < self.best_val_loss:
                self.best_val_f1 = max(self.best_val_f1, val_metrics["f1_score"])
                self.best_val_loss = min(self.best_val_loss, val_metrics["loss"])
                self.patience_counter = 0

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "val_metrics": val_metrics,
                    "epoch": epoch
                }, save_path_obj)
                logger.info(f"--> Saved best model checkpoint to {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}!")
                    break

        elapsed = time.time() - start_time
        logger.info(f"Training for '{self.model_name}' completed in {elapsed:.2f} seconds.")
        return history
