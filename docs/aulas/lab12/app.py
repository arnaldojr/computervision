import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import MobileNetTransfer


app = FastAPI(
    title="Transfer Learning API — CIFAR-10",
    description="API didática para classificação com MobileNetV3 fine-tuned no CIFAR-10.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mobilenet_cifar10.pt"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
FRONTEND_PATH = BASE_DIR / "frontend.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: MobileNetTransfer | None = None
METADATA: dict[str, Any] = {}

DEFAULT_METADATA: dict[str, Any] = {
    "model_name": "MobileNetV3-Small (Transfer Learning)",
    "dataset": "CIFAR-10",
    "input_size": 224,
    "classes": ["aviao", "automovel", "passaro", "gato", "cervo",
                 "cachorro", "sapo", "cavalo", "navio", "caminhao"],
    "normalization": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
}


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return DEFAULT_METADATA


def load_model(metadata: dict[str, Any]) -> MobileNetTransfer:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o notebook_transfer_learning.ipynb e copie a pasta artifacts/."
        )
    num_classes = len(metadata.get("classes", []) or []) or 10
    model = MobileNetTransfer(num_classes=num_classes, freeze_backbone=False).to(DEVICE)
    model.backbone.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, METADATA
    METADATA = load_metadata()
    MODEL = load_model(METADATA)


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


def preprocess_image(image: Image.Image) -> torch.Tensor:
    meta = METADATA.get("normalization", DEFAULT_METADATA["normalization"])
    size = METADATA.get("input_size", 224)
    pipeline = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["mean"], std=meta["std"]),
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

    classes = METADATA.get("classes", [])
    top_probs, top_indices = torch.topk(probs, k=3)
    top3 = [
        {"classe_id": int(i), "classe": classes[int(i)] if int(i) < len(classes) else str(int(i)),
         "probabilidade": round(float(p), 4)}
        for i, p in zip(top_indices, top_probs)
    ]
    return {"classe_prevista": top3[0]["classe"], "classe_id": top3[0]["classe_id"],
            "probabilidade": top3[0]["probabilidade"], "top3": top3}
