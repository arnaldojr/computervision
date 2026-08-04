import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from model import DeiTClassifier


app = FastAPI(
    title="Vision Transformer API — CIFAR-10",
    description="API didática para classificação com DeiT-Tiny fine-tuned no CIFAR-10.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"
FRONTEND_PATH = BASE_DIR / "frontend.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: DeiTClassifier | None = None
METADATA: dict[str, Any] = {}

DEFAULT_METADATA: dict[str, Any] = {
    "model_name": "DeiT-Tiny (fine-tuned CIFAR-10)",
    "base_model": "facebook/deit-tiny-patch16-224",
    "dataset": "CIFAR-10",
    "classes": ["aviao", "automovel", "passaro", "gato", "cervo",
                 "cachorro", "sapo", "cavalo", "navio", "caminhao"],
}


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return DEFAULT_METADATA


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, METADATA
    METADATA = load_metadata()
    classes = METADATA.get("classes", DEFAULT_METADATA["classes"])
    MODEL = DeiTClassifier(num_classes=len(classes), device=DEVICE)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=500, detail="frontend.html não encontrado.")
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "modelo_carregado": MODEL is not None,
        "modelo": METADATA.get("model_name"),
        "classes": METADATA.get("classes"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida.") from exc

    # o processor do HuggingFace cuida de resize e normalização
    inputs = MODEL.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        logits = MODEL(pixel_values)
        probs = torch.softmax(logits, dim=1)[0]

    classes = METADATA.get("classes", DEFAULT_METADATA["classes"])
    top_probs, top_indices = torch.topk(probs, k=3)
    top3 = [
        {"classe_id": int(i), "classe": classes[int(i)] if int(i) < len(classes) else str(int(i)),
         "probabilidade": round(float(p), 4)}
        for i, p in zip(top_indices, top_probs)
    ]
    return {"classe_prevista": top3[0]["classe"], "classe_id": top3[0]["classe_id"],
            "probabilidade": top3[0]["probabilidade"], "top3": top3}
