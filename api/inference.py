import io
import time
import base64
import torch
import numpy as np

from PIL import Image

from utils import preprocess, decode_segmap, overlay_mask
from model_loader import model_manager


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_inference(image_bytes, model_name, overlay_alpha=0.5):

    start = time.time()

    # -----------------------------
    # Load image
    # -----------------------------
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    # -----------------------------
    # Preprocess
    # -----------------------------
    pixel_values = preprocess(image).to(DEVICE)

    # -----------------------------
    # Select model
    # -----------------------------
    model = model_manager.get_model(model_name)
    model.eval()

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():

        with torch.amp.autocast("cuda"):

            # normal prediction
            out1 = model(pixel_values=pixel_values)
            logits1 = out1.logits

            # flipped prediction
            flipped = torch.flip(pixel_values, dims=[3])
            out2 = model(pixel_values=flipped)

            logits2 = torch.flip(out2.logits, dims=[3])

            # average predictions
            logits = (logits1 + logits2) / 2

        preds = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # -----------------------------
    # Resize mask back
    # -----------------------------
    preds = np.array(
        Image.fromarray(preds.astype(np.uint8)).resize((w, h), Image.NEAREST)
    )

    # -----------------------------
    # Convert to colored mask
    # -----------------------------
    color_mask = decode_segmap(preds)

    # -----------------------------
    # Overlay
    # -----------------------------
    overlay = overlay_mask(image, color_mask, alpha=overlay_alpha)

    # -----------------------------
    # Convert to base64
    # -----------------------------
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")

    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    latency = int((time.time() - start) * 1000)

    return overlay_b64, latency