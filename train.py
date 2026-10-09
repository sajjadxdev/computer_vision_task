import argparse
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data import prepare_dataset, create_dataloaders
from src.models import build_baseline_model, build_transfer_model
from src.training import Trainer

logger = setup_logger("MasterTrainPipeline")


def run_training_pipeline(config_path: str = "config/config.yaml"):
    """
    Executes complete master training pipeline:
    1. Prepares & splits dataset (train 70%, val 15%, test 15%).
    2. Builds PyTorch DataLoaders with domain augmentations.
    3. Trains Baseline Custom CNN (Model 1).
    4. Fine-tunes Transfer Learning MobileNetV2 (Model 2).
    5. Saves best model checkpoints to artifacts/models/.
    """
    logger.info("==================================================")
    logger.info("  STARTING MASTER TRAINING PIPELINE (Fire & Smoke)")
    logger.info("==================================================")

    config = load_config(config_path)
    set_seed(config["data"]["seed"])

    # Step 1: Dataset Preparation
    prepare_dataset(config_path)

    # Step 2: DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Step 3: Train Model 1 (Baseline CNN)
    logger.info("\n--- STEP 1/2: Training Model 1 (Baseline Custom CNN) ---")
    baseline_model = build_baseline_model(config)
    baseline_trainer = Trainer(baseline_model, config, model_name="Baseline_CNN")
    baseline_history = baseline_trainer.train(
        train_loader,
        val_loader,
        save_path=config["models"]["baseline_cnn"]["checkpoint_path"]
    )

    # Step 4: Train Model 2 (Transfer Learning MobileNetV2)
    logger.info("\n--- STEP 2/2: Training Model 2 (Transfer Learning MobileNetV2) ---")
    transfer_model = build_transfer_model(config)
    transfer_trainer = Trainer(transfer_model, config, model_name="Transfer_MobileNetV2")
    transfer_history = transfer_trainer.train(
        train_loader,
        val_loader,
        save_path=config["models"]["transfer_learning"]["checkpoint_path"]
    )

    logger.info("\n==================================================")
    logger.info("  MASTER TRAINING PIPELINE SUCCESSFULLY FINISHED!")
    logger.info("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fire & Smoke CV Models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    run_training_pipeline(args.config)
