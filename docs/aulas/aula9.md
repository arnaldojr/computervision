## Deploy de Modelos de Visão Computacional

Material da Aula 9 — Deploy de modelos com FastAPI.
Nesta aula você vai carregar um modelo pré-treinado e servi-lo como uma API REST, testando via Postman e interface web.

[Lab 11 — Deploy FastAPI (download)](lab11/lab11.zip){ .md-button .md-button--primary }

---

### Por que fazer deploy de modelos?

Um modelo treinado que fica apenas no notebook não gera valor. **Deploy** é o processo de disponibilizar o modelo para que outros sistemas — ou pessoas — possam usá-lo.

```
Notebook → artefatos/ → API → cliente (web / Postman / app)
```

!!! tip "Vocabulário"
    - **Artefato:** arquivo do modelo salvo (`modelo.pt`, `modelo.pkl`)
    - **Endpoint:** URL que recebe requisições (`POST /predict`)
    - **Inferência:** executar o modelo sobre novos dados

---

### Arquitetura do lab

```
┌───────────────┐        POST /predict        ┌──────────────────────┐
│  Cliente      │  ──── imagem (multipart) ──► │  FastAPI (app.py)    │
│  (browser /   │                              │  ├── model.py        │
│   Postman)    │  ◄── JSON { classe, top3 } ──│  └── artifacts/      │
└───────────────┘                              └──────────────────────┘
```

- **`model.py`** — define e carrega o modelo
- **`app.py`** — define os endpoints e a lógica de pré-processamento
- **`artifacts/`** — armazena os pesos e metadados do modelo
- **`frontend.html`** — interface web servida pela própria API

---

### Modelos pré-treinados com torchvision

O `torchvision` disponibiliza dezenas de modelos já treinados no ImageNet (1000 classes). Você pode usá-los diretamente sem nenhum treinamento.

=== "Carregar modelo"

    ```python
    import torchvision.models as models

    # MobileNetV3-Small: leve, rápido, boa acurácia
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.eval()
    ```

=== "Pré-processamento obrigatório"

    Cada modelo espera uma normalização específica. Para modelos ImageNet:

    ```python
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    ```

=== "Comparativo de modelos"

    | Modelo | Parâmetros | Top-1 ImageNet | Latência (CPU) |
    |--------|-----------|----------------|----------------|
    | MobileNetV3-Small | 2.5M | 67.7% | ~10ms |
    | MobileNetV3-Large | 5.5M | 75.3% | ~20ms |
    | ResNet-18 | 11.7M | 69.8% | ~30ms |
    | ResNet-50 | 25.6M | 76.1% | ~60ms |

---

### FastAPI em 5 minutos

!!! info "Por que FastAPI?"
    - Documentação interativa automática (`/docs`)
    - Validação de tipos com Python type hints
    - Suporte nativo a upload de arquivos
    - Alta performance (baseado em Starlette + Pydantic)

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # lê bytes → PIL Image → tensor → modelo → JSON
    ...
```

Iniciar o servidor:

```bash
uvicorn app:app --reload
```

Acessar a documentação interativa: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Estrutura do lab11

```
lab11/
├── app.py                  # servidor FastAPI
├── model.py                # carregamento do modelo
├── frontend.html           # interface web
├── requirements.txt        # dependências
├── notebook_deploy.ipynb   # notebook com inferência local e testes
└── artifacts/
    └── metadata.json       # classes ImageNet e config do modelo
```

---

### Como executar

```bash
# 1. instalar dependências
pip install -r requirements.txt

# 2. iniciar a API
uvicorn app:app --reload

# 3. abrir no navegador
# http://localhost:8000          → interface web
# http://localhost:8000/docs     → Swagger UI (Postman alternativo)
```

!!! warning "Primeira execução"
    O torchvision faz o download dos pesos do modelo (~10MB) na primeira vez que `model.py` é importado. É necessária conexão com a internet.

---

### Testando com Postman

1. Método: `POST`
2. URL: `http://localhost:8000/predict`
3. Body → `form-data` → Key: `file` (tipo File) → selecionar uma imagem
4. Enviar e verificar o JSON de resposta

Resposta esperada:

```json
{
  "classe_prevista": "tabby cat",
  "classe_id": 281,
  "probabilidade": 0.8731,
  "top3": [
    { "classe": "tabby cat", "classe_id": 281, "probabilidade": 0.8731 },
    { "classe": "tiger cat", "classe_id": 282, "probabilidade": 0.0912 },
    { "classe": "Egyptian cat", "classe_id": 285, "probabilidade": 0.0241 }
  ]
}
```

---

### Referências

- [FastAPI — documentação oficial](https://fastapi.tiangolo.com/)
- [torchvision — model zoo](https://pytorch.org/vision/stable/models.html)
- [ImageNet classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
- [uvicorn](https://www.uvicorn.org/)
