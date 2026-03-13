import torch
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from inference import run_inference
from model_loader import model_manager


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SegFormer Multi-Terrain API")


# ----------------------------------------------------
# CORS (Frontend Access)
# ----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Load Models On Startup
# ----------------------------------------------------

@app.on_event("startup")
def load_models():

    print("Loading SegFormer models...")

    model_manager.load_model(
        "desert",
        "../checkpoints/desert_segformer_b2/best_model",
        num_classes=6
    )

    model_manager.load_model(
        "mountain",
        "../checkpoints/mountain_forest_segformer_b2/best_model",
        num_classes=15
    )

    model_manager.load_model(
        "roads",
        "../checkpoints/roads_segformer_b2/best_model",
        num_classes=20
    )

    print("All models loaded successfully.")


# ----------------------------------------------------
# Health Check
# ----------------------------------------------------

@app.get("/")
def root():
    return {"message": "SegFormer Multi-Terrain API Running"}


# ----------------------------------------------------
# Inference Endpoint
# ----------------------------------------------------

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model: str = Query("desert"),
    overlay_alpha: float = 0.5
):

    image_bytes = await file.read()

    overlay_b64, latency = run_inference(
        image_bytes,
        model,
        overlay_alpha
    )

    return {
        "overlay_png_b64": overlay_b64,
        "latency_ms": latency
    }