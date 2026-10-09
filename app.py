import uvicorn
from src.inference.api import app
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    host = config["inference"].get("host", "0.0.0.0")
    port = config["inference"].get("port", 8000)

    print(f"Starting Fire & Smoke Computer Vision Deployment Server on http://{host}:{port}")
    uvicorn.run("src.inference.api:app", host=host, port=port, reload=False)
