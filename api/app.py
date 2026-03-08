import time
import base64
import io
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from utils import preprocess, decode_segmap, overlay_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6

app = FastAPI()

# CORS (frontend support)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "../segformer/checkpoints/best_model",
    num_labels=NUM_CLASSES
).to(DEVICE)
model.eval()

@app.post("/infer")
async def infer(file: UploadFile = File(...), overlay_alpha: float = 0.5):
    start = time.time()

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    w, h = image.size

    pixel_values = preprocess(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        preds = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()

    preds = np.array(
        Image.fromarray(preds.astype(np.uint8)).resize((w, h), Image.NEAREST)
    )

    color_mask = decode_segmap(preds)
    overlay = overlay_mask(image, color_mask, alpha=overlay_alpha)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "overlay_png_b64": b64,
        "latency_ms": int((time.time() - start) * 1000)
    }
