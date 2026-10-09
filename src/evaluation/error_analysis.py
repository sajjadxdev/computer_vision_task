import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger("ErrorAnalysis")


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_path: str):
    """Generates and saves a polished confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix - Fire & Smoke Classifier", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Ground Truth Class", fontsize=12)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def plot_roc_curve(roc_data: Dict[str, List[float]], roc_auc: float, save_path: str):
    """Generates and saves ROC Curve plot."""
    plt.figure(figsize=(7, 6))
    plt.plot(roc_data["fpr"], roc_data["tpr"], color="#d9534f", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve plot to {save_path}")


def visualize_error_samples(
    misclassified_samples: List[Dict[str, Any]],
    class_names: List[str],
    save_path: str,
    max_samples: int = 6
):
    """
    Plots a visual grid of misclassified images for qualitative error analysis.
    Shows True vs Predicted class, confidence score, and file path.
    """
    if not misclassified_samples:
        logger.info("No misclassified samples found! Model achieved 100% accuracy on test set.")
        return

    n_samples = min(len(misclassified_samples), max_samples)
    cols = 3
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_samples):
        sample = misclassified_samples[idx]
        img_path = sample["path"]
        true_cls = class_names[sample["true_label"]]
        pred_cls = class_names[sample["pred_label"]]
        conf = sample["confidence"]

        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"True: {true_cls} | Pred: {pred_cls}\nConf: {conf:.2%}",
                color="crimson",
                fontsize=11,
                fontweight="bold"
            )
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Image Error\n{e}", ha="center", va="center")

        axes[idx].axis("off")

    # Hide extra unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Qualitative Error Analysis: Misclassified Test Samples", fontsize=16, y=1.02, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error analysis visualization to {save_path}")
