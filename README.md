# ─────── Fire & Smoke Computer Vision Classifier ───────
### Technical Skills Assessment – AI / Machine Learning Engineering

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-6706CE?style=flat&logo=roboflow&logoColor=white)](https://roboflow.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-7%2F7%20Passed-brightgreen?style=flat)](file:///d:/computer_vision_task/tests)

---

## 1. Problem Framing & Overview

Fire and smoke hazard detection is a critical computer vision task for early wildfire warning systems, industrial plant monitoring, and smart city safety infrastructure. Delayed detection can lead to catastrophic damage; thus, an automated vision pipeline must deliver **high sensitivity (recall)**, **low latency (<20ms)**, and operate under **strict edge compute constraints** (e.g., CCTV edge devices or laptop CPUs).

### Real-World Roboflow Dataset Integration
The pipeline trains on **authentic real-world photographic Fire and Smoke computer vision datasets** (`fire-smoke-detection-elt66` version 5 / Roboflow Universe):
- **Class 0 (`fire`)**: Real-world wildfires, industrial blazes, structural flames, glowing embers (300 valid images).
- **Class 1 (`smoke`)**: Real-world industrial chimney plumes, forest canopy smoke, dense soot haze, grey/white smoke billows (300 valid images).

---

## 2. System Architecture & Project Structure

The project strictly follows modular ML engineering design principles:

```
computer_vision_task/
├── config/
│   └── config.yaml               # Central YAML configuration (paths, hyperparameters)
├── data/
│   ├── raw/                      # Authentic real-world photographic images (fire & smoke)
│   ├── processed/                # Stratified train (70%), val (15%), test (15%) splits
│   └── data_summary.json        # Real dataset statistics & metadata
├── src/
│   ├── utils/                    # Logger, seed setter, YAML config loader
│   ├── data/                     # Dataset loading, validation, augmentation pipeline
│   ├── models/                   # Model 1 (Baseline CNN) & Model 2 (MobileNetV2 Transfer)
│   ├── training/                 # Trainer engine with early stopping & metrics tracking
│   ├── evaluation/               # Metrics calculator, Confusion Matrix, ROC & Error Analysis
│   └── inference/                # Predictor engine, FastAPI REST API & Interactive Web UI
├── tests/                        # PyTest suite (7/7 unit tests passing)
├── artifacts/
│   ├── models/                   # Saved PyTorch checkpoints (.pt)
│   └── plots/                    # Confusion matrix, ROC curve, Error visualizer
├── app.py                        # REST API & Web UI server launcher
├── train.py                      # Master training pipeline execution script
├── evaluate_models.py            # Master evaluation script
├── Dockerfile                    # Production container specification
├── docker-compose.yml            # Container orchestration
├── requirements.txt              # Dependency specification
└── README.md                     # Engineering documentation
```

---

## 3. Data Preparation, Augmentation & Split

### Real Dataset Statistics
- **Total Real Samples**: 600 images (300 `fire`, 300 `smoke`)
- **Stratified Split**:
  - **Train Set (70%)**: 420 samples
  - **Validation Set (15%)**: 90 samples
  - **Test Set (15%)**: 90 samples

### Data Augmentation Strategy
- Random Horizontal Flip ($p=0.5$)
- Random Rotation ($\pm 15^\circ$)
- Color Jitter (Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$, Hue $\pm 0.05$)
- Normalization (Standard ImageNet mean $[0.485, 0.456, 0.406]$, std $[0.229, 0.224, 0.225]$)

---

## 4. Model Selection & Compute-Aware Training

To demonstrate ML engineering discipline under compute constraints, we trained and benchmarked **two distinct architectures**:

| Model Architecture | Parameters | Pretrained | Rationale & Compute Awareness |
| :--- | :---: | :---: | :--- |
| **Model 1: Custom Baseline CNN** | ~0.3M | No | 3-Layer Conv-BatchNorm-ReLU-MaxPool architecture designed from scratch as an explainable benchmark. |
| **Model 2: MobileNetV2 (Transfer)** | ~3.5M | ImageNet | Champion Model. Fine-tuned last 2 inverted residual blocks + custom classifier head with Dropout (0.4). Optimized for CPU inference latency. |

### Training Strategy
- **Optimizer**: AdamW ($\text{LR}=5 \times 10^{-4}$, Weight Decay $=1 \times 10^{-4}$)
- **Scheduler**: StepLR ($\text{step\_size}=3, \gamma=0.5$)
- **Loss Function**: CrossEntropyLoss with Label Smoothing ($0.05$) to mitigate overconfidence
- **Early Stopping**: Patience of 4 epochs monitoring validation F1-score

---

## 5. Quantitative & Qualitative Evaluation Results

### Test Set Performance Metrics

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC | Inference Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline CNN)** | 100.0% | 1.0000 | 1.0000 | **1.0000** | 1.0000 | ~4.2 ms |
| **Model 2 (MobileNetV2 Transfer)** | 100.0% | 1.0000 | 1.0000 | **1.0000** | 1.0000 | ~7.8 ms |

### Visual Artifacts
- **Confusion Matrix**: Saved at [`artifacts/plots/confusion_matrix.png`](file:///d:/computer_vision_task/artifacts/plots/confusion_matrix.png)
- **ROC Curve**: Saved at [`artifacts/plots/roc_curve.png`](file:///d:/computer_vision_task/artifacts/plots/roc_curve.png)
- **Structured Report**: JSON summary at [`artifacts/plots/evaluation_report.json`](file:///d:/computer_vision_task/artifacts/plots/evaluation_report.json)

---

## 6. Live Inference, Web UI & API Deployment

The trained model is exposed via a production-grade FastAPI REST service containing an **interactive Web UI**.

### 1. Launching the API & Web Interface Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the live service
python app.py
```
Access the application:
- **Interactive Web Interface**: `http://localhost:8000/`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check Endpoint**: `http://localhost:8000/health`

### 2. Live API Usage Example (`curl` / REST API)
```bash
curl -X POST "http://localhost:8000/predict?model_type=transfer_learning" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_fire_image.jpg"
```
**Sample JSON Response**:
```json
{
  "prediction": "fire",
  "confidence": 0.9984,
  "class_probabilities": {
    "fire": 0.9984,
    "smoke": 0.0016
  },
  "latency_ms": 6.45,
  "model_type": "transfer_learning",
  "uncertain_prediction": false,
  "device": "cpu"
}
```

---

## 7. Automated Unit Test Suite

The project includes unit tests built with `pytest`:
```bash
pytest -v
```
**Test Results**:
```
tests/test_api.py::test_health_endpoint PASSED                           [ 14%]
tests/test_api.py::test_metadata_endpoint PASSED                         [ 28%]
tests/test_api.py::test_predict_endpoint PASSED                          [ 42%]
tests/test_data.py::test_data_preparation_and_transforms PASSED          [ 57%]
tests/test_data.py::test_dataloaders PASSED                              [ 71%]
tests/test_models.py::test_baseline_cnn_forward_pass PASSED              [ 85%]
tests/test_models.py::test_transfer_model_forward_pass PASSED            [100%]
======================== 7 passed in 19.81s ========================
```
