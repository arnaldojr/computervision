import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import ModeloPreTreinado


app = FastAPI(
    title="Deploy API — Visão Computacional",
    description="API didática para inferência com modelo pré-treinado no ImageNet.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
FRONTEND_PATH = BASE_DIR / "frontend.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL: ModeloPreTreinado | None = None
METADATA: dict[str, Any] = {}


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "model_name": "MobileNetV3-Small",
        "input_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, METADATA
    METADATA = load_metadata()
    MODEL = ModeloPreTreinado(device=DEVICE)


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
    }


def preprocess_image(image: Image.Image) -> torch.Tensor:
    meta = METADATA.get("normalization", {})
    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])
    size = METADATA.get("input_size", 224)

    pipeline = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return pipeline(image.convert("RGB")).unsqueeze(0).to(DEVICE)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    raw = await file.read()
    try:
        image = Image.open(BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida.") from exc

    tensor = preprocess_image(image)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=3)
    classes = MODEL.classes

    top3 = [
        {
            "classe_id": int(idx.item()),
            "classe": classes[int(idx.item())],
            "probabilidade": round(float(p.item()), 4),
        }
        for p, idx in zip(top_probs, top_indices)
    ]

    return {
        "classe_prevista": top3[0]["classe"],
        "classe_id": top3[0]["classe_id"],
        "probabilidade": top3[0]["probabilidade"],
        "top3": top3,
    }
