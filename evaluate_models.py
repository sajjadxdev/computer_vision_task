import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data import create_dataloaders
from src.models import build_baseline_model, build_transfer_model
from src.evaluation import compute_metrics, plot_confusion_matrix, plot_roc_curve, visualize_error_samples

logger = setup_logger("MasterEvaluation")


def evaluate_model_on_test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    """Executes full evaluation on test split and collects ground truth, predictions, and probabilities."""
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []
    y_prob = []
    sample_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1] # Probability of positive class (smoke)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            sample_paths.extend(paths)

    return y_true, y_pred, y_prob, sample_paths


def run_evaluation_pipeline(config_path: str = "config/config.yaml"):
    """
    Executes master evaluation pipeline:
    1. Loads test split data.
    2. Evaluates Model 1 (Baseline CNN) and Model 2 (Transfer Learning).
    3. Computes comparative metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
    4. Generates Confusion Matrix, ROC Curve, and Error Analysis Grid plots.
    5. Saves structured evaluation report JSON.
    """
    logger.info("==================================================")
    logger.info("  STARTING MASTER EVALUATION & ERROR ANALYSIS")
    logger.info("==================================================")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = config["data"]["classes"]
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = create_dataloaders(config)

    # 1. Evaluate Model 1 (Baseline CNN)
    logger.info("\nEvaluating Model 1 (Baseline Custom CNN)...")
    baseline_model = build_baseline_model(config)
    b_ckpt_path = Path(config["models"]["baseline_cnn"]["checkpoint_path"])
    if b_ckpt_path.exists():
        ckpt = torch.load(b_ckpt_path, map_location=device)
        baseline_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    
    b_true, b_pred, b_prob, b_paths = evaluate_model_on_test(baseline_model, test_loader, device)
    b_metrics = compute_metrics(b_true, b_pred, b_prob, class_names)
    logger.info(f"Model 1 (Baseline CNN) Test F1-Score: {b_metrics['macro_f1_score']:.4f} | Acc: {b_metrics['accuracy']:.4f}")

    # 2. Evaluate Model 2 (Transfer MobileNetV2)
    logger.info("\nEvaluating Model 2 (Transfer Learning MobileNetV2)...")
    transfer_model = build_transfer_model(config)
    t_ckpt_path = Path(config["models"]["transfer_learning"]["checkpoint_path"])
    if t_ckpt_path.exists():
        ckpt = torch.load(t_ckpt_path, map_location=device)
        transfer_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    
    t_true, t_pred, t_prob, t_paths = evaluate_model_on_test(transfer_model, test_loader, device)
    t_metrics = compute_metrics(t_true, t_pred, t_prob, class_names)
    logger.info(f"Model 2 (Transfer MobileNetV2) Test F1-Score: {t_metrics['macro_f1_score']:.4f} | Acc: {t_metrics['accuracy']:.4f}")

    # Plot Visual Artifacts for Champion Model (Model 2)
    logger.info("\nGenerating Visual Artifacts and Qualitative Error Analysis...")
    plot_confusion_matrix(
        t_metrics["confusion_matrix"],
        class_names,
        str(output_dir / config["evaluation"]["confusion_matrix_filename"])
    )
    plot_roc_curve(
        t_metrics["roc_curve"],
        t_metrics["roc_auc"],
        str(output_dir / config["evaluation"]["roc_curve_filename"])
    )

    # Collect Misclassified Samples for Error Analysis Grid
    misclassified = []
    for idx in range(len(t_true)):
        if t_true[idx] != t_pred[idx]:
            misclassified.append({
                "path": t_paths[idx],
                "true_label": t_true[idx],
                "pred_label": t_pred[idx],
                "confidence": max(t_prob[idx], 1.0 - t_prob[idx])
            })

    visualize_error_samples(
        misclassified,
        class_names,
        str(output_dir / config["evaluation"]["error_analysis_filename"])
    )

    # Save Comparative Evaluation Report JSON
    report = {
        "dataset_summary": {
            "test_samples": len(b_true),
            "classes": class_names
        },
        "model_comparison": {
            "baseline_cnn": {
                "accuracy": b_metrics["accuracy"],
                "macro_precision": b_metrics["macro_precision"],
                "macro_recall": b_metrics["macro_recall"],
                "macro_f1_score": b_metrics["macro_f1_score"],
                "roc_auc": b_metrics["roc_auc"],
                "per_class": b_metrics["per_class"]
            },
            "transfer_learning_mobilenet_v2": {
                "accuracy": t_metrics["accuracy"],
                "macro_precision": t_metrics["macro_precision"],
                "macro_recall": t_metrics["macro_recall"],
                "macro_f1_score": t_metrics["macro_f1_score"],
                "roc_auc": t_metrics["roc_auc"],
                "per_class": t_metrics["per_class"]
            }
        },
        "error_analysis_summary": {
            "total_misclassifications": len(misclassified),
            "error_rate": len(misclassified) / max(len(b_true), 1)
        }
    }

    report_path = output_dir / config["evaluation"]["evaluation_report_filename"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report successfully saved to {report_path}")
    logger.info("==================================================")
    logger.info("  MASTER EVALUATION & ERROR ANALYSIS COMPLETE!")
    logger.info("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fire & Smoke Models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    run_evaluation_pipeline(args.config)
