import json
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import RedeNeuralSimples


app = FastAPI(
    title="MNIST API - PyTorch + FastAPI",
    description="API didática para classificar dígitos manuscritos usando um modelo treinado no notebook.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "../artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mnist_mlp.pt"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
FRONTEND_PATH = BASE_DIR / "frontend.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL: RedeNeuralSimples | None = None

DEFAULT_METADATA: dict[str, Any] = {
    "model_name": "RedeNeuralSimples",
    "input_shape": [1, 28, 28],
    "classes": list(range(10)),
    "normalization": {
        "mean": 0.1307,
        "std": 0.3081,
    },
}

METADATA: dict[str, Any] = DEFAULT_METADATA.copy()


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return DEFAULT_METADATA


def load_model() -> RedeNeuralSimples:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute primeiro o notebook e copie a pasta artifacts para este projeto."
        )

    model = RedeNeuralSimples().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


@app.on_event("startup")
def startup_event() -> None:
    global MODEL
    global METADATA

    METADATA = load_metadata()
    MODEL = load_model()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """
    Página simples para desenhar um dígito e enviar para a API.
    """
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=500, detail="Arquivo frontend.html não encontrado.")

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
    """
    Converte uma imagem comum em tensor no mesmo formato usado no treino.

    A interface web envia um desenho branco em fundo preto, que já é parecido
    com o padrão visual do MNIST.
    """
    image = image.convert("L").resize((28, 28))

    tensor = transforms.ToTensor()(image)

    # MNIST normalmente tem fundo escuro e dígito claro.
    # Se a imagem enviada tiver fundo claro e dígito escuro, invertemos.
    if float(tensor.mean()) > 0.5:
        tensor = 1.0 - tensor

    mean = METADATA["normalization"]["mean"]
    std = METADATA["normalization"]["std"]

    tensor = transforms.Normalize((mean,), (std,))(tensor)

    # De [1, 28, 28] para [1, 1, 28, 28]
    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    raw = await file.read()

    try:
        image = Image.open(BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado não é uma imagem válida.",
        ) from exc

    x = preprocess_image(image)

    with torch.no_grad():
        logits = MODEL(x)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())

    top_probs, top_indices = torch.topk(probabilities, k=3)

    top3 = [
        {
            "classe": int(index.item()),
            "probabilidade": round(float(prob.item()), 4),
        }
        for prob, index in zip(top_probs, top_indices)
    ]

    return {
        "arquivo": file.filename,
        "classe_prevista": predicted_class,
        "probabilidade_classe_prevista": round(float(probabilities[predicted_class].item()), 4),
        "top3": top3,
    }
