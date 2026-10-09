import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference.predictor import FireSmokePredictor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("FastAPI_Deployment")
config = load_config("config/config.yaml")

app = FastAPI(
    title="Fire & Smoke Computer Vision API",
    description="REST API and Web UI for real-time Fire and Smoke detection & classification using PyTorch deep learning models.",
    version="1.0.0"
)

# Enable CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor_instance = None


def get_predictor(model_type: str = "transfer_learning") -> FireSmokePredictor:
    global predictor_instance
    if predictor_instance is None or predictor_instance.model_type != model_type:
        predictor_instance = FireSmokePredictor(model_type=model_type)
    return predictor_instance


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Fire & Smoke Computer Vision API service...")
    # Warm up default predictor
    get_predictor(config["inference"].get("model_to_use", "transfer_learning"))


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for container orchestrators and cloud load balancers."""
    return {
        "status": "healthy",
        "service": config["project"]["name"],
        "version": config["project"]["version"],
        "classes": config["data"]["classes"]
    }


@app.get("/metadata", tags=["Metadata"])
async def get_metadata():
    """Returns model and dataset configuration metadata."""
    return {
        "project": config["project"],
        "data": config["data"],
        "models_available": list(config["models"].keys()),
        "active_model": config["inference"]["model_to_use"]
    }


@app.post("/predict", tags=["Inference"])
async def predict_image(
    file: UploadFile = File(...),
    model_type: str = Query("transfer_learning", description="Model architecture: 'transfer_learning' or 'baseline_cnn'")
):
    """
    Exposes real-time image prediction REST API.
    Accepts JPG, PNG, JPEG binary image file via multipart upload.
    Returns JSON response with prediction, confidence score, class probabilities, and latency.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload an image file (JPG, PNG, JPEG).")

    try:
        contents = await file.read()
        predictor = get_predictor(model_type=model_type)
        result = predictor.predict(contents)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_web_ui():
    """Serves a modern, responsive web UI for drag-and-drop image prediction."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fire & Smoke Computer Vision AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
            body { background: #0f172a; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
            .container { max-width: 850px; width: 100%; background: #1e293b; border-radius: 16px; padding: 2.5rem; box-shadow: 0 20px 40px rgba(0,0,0,0.5); border: 1px solid #334155; }
            h1 { font-size: 2.2rem; margin-bottom: 0.5rem; text-align: center; background: linear-gradient(135deg, #f97316, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p.subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; }
            .upload-box { border: 2px dashed #475569; border-radius: 12px; padding: 2.5rem; text-align: center; cursor: pointer; transition: all 0.3s ease; background: #0f172a; }
            .upload-box:hover { border-color: #f97316; background: #1e293b; }
            .drop-text { color: #94a3b8; font-size: 1.1rem; margin-top: 1rem; }
            input[type="file"] { display: none; }
            .model-select { margin: 1.5rem 0; display: flex; justify-content: center; gap: 1rem; align-items: center; }
            select { background: #0f172a; color: #f8fafc; border: 1px solid #475569; padding: 0.6rem 1.2rem; border-radius: 8px; font-size: 1rem; }
            .btn { background: linear-gradient(135deg, #ea580c, #dc2626); color: #fff; border: none; padding: 0.8rem 2rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: 0.2s ease; }
            .btn:hover { opacity: 0.9; transform: translateY(-1px); }
            #preview { max-width: 100%; max-height: 280px; border-radius: 10px; margin-top: 1.5rem; display: none; margin-left: auto; margin-right: auto; }
            .result-card { margin-top: 2rem; background: #0f172a; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; display: none; }
            .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
            .badge-fire { background: #ef4444; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-size: 1rem; }
            .badge-smoke { background: #64748b; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-size: 1rem; }
            .prob-bar-container { margin-bottom: 0.8rem; }
            .prob-label { display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 0.3rem; color: #cbd5e1; }
            .prob-bar { height: 10px; background: #334155; border-radius: 5px; overflow: hidden; }
            .prob-fill { height: 100%; background: linear-gradient(90deg, #f97316, #ef4444); width: 0%; transition: width 0.5s ease; }
            .latency { text-align: right; color: #64748b; font-size: 0.85rem; margin-top: 1rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fire & Smoke Computer Vision AI</h1>
            <p class="subtitle">Deployable Deep Learning Model for Real-Time Hazard Detection</p>

            <div class="model-select">
                <label for="modelChoice">Select Model Architecture:</label>
                <select id="modelChoice">
                    <option value="transfer_learning" selected>Model 2: Transfer Learning (MobileNetV2)</option>
                    <option value="baseline_cnn">Model 1: Custom Baseline CNN</option>
                </select>
            </div>

            <div class="upload-box" id="dropArea" onclick="document.getElementById('fileInput').click()">
                <svg width="48" height="48" fill="none" stroke="#f97316" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <div class="drop-text">Click or Drag & Drop image here to classify</div>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                <img id="preview" alt="Image Preview">
            </div>

            <div style="text-align: center; margin-top: 1.5rem;">
                <button class="btn" onclick="uploadAndPredict()">Run AI Prediction</button>
            </div>

            <div class="result-card" id="resultCard">
                <div class="result-title">
                    <span>Classification Output</span>
                    <span id="predBadge" class="badge-fire">FIRE</span>
                </div>
                <div id="probBars"></div>
                <div class="latency" id="latencyText">Latency: 0 ms</div>
            </div>
        </div>

        <script>
            let selectedFile = null;

            function handleFileSelect(e) {
                const files = e.target.files;
                if (files && files[0]) {
                    selectedFile = files[0];
                    const reader = new FileReader();
                    reader.onload = function(evt) {
                        const preview = document.getElementById('preview');
                        preview.src = evt.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(selectedFile);
                }
            }

            async function uploadAndPredict() {
                if (!selectedFile) {
                    alert("Please select or drop an image file first.");
                    return;
                }

                const formData = new FormData();
                formData.append("file", selectedFile);

                const modelType = document.getElementById('modelChoice').value;
                const resultCard = document.getElementById('resultCard');

                try {
                    const response = await fetch(`/predict?model_type=${modelType}`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error("Prediction API failed");
                    }

                    const data = await response.json();
                    
                    // Display results
                    resultCard.style.display = 'block';
                    const badge = document.getElementById('predBadge');
                    badge.innerText = `${data.prediction.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`;
                    badge.className = data.prediction === 'fire' ? 'badge-fire' : 'badge-smoke';

                    // Probabilities Bars
                    const probContainer = document.getElementById('probBars');
                    probContainer.innerHTML = '';
                    for (const [cls, prob] of Object.entries(data.class_probabilities)) {
                        const pct = (prob * 100).toFixed(1);
                        probContainer.innerHTML += `
                            <div class="prob-bar-container">
                                <div class="prob-label"><span>${cls.toUpperCase()}</span><span>${pct}%</span></div>
                                <div class="prob-bar"><div class="prob-fill" style="width: ${pct}%"></div></div>
                            </div>
                        `;
                    }

                    document.getElementById('latencyText').innerText = `Inference Latency: ${data.latency_ms} ms | Device: ${data.device}`;
                } catch (err) {
                    alert("Error executing inference: " + err.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
