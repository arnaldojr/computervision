# Parte 2 — PyTorch + FastAPI + Frontend (CIFAR-10)

Vamos usar o modelo treinado no notebook como motor de uma API.

## Fluxo da aplicação

```text
usuario seleciona uma imagem no navegador
        ↓
frontend envia arquivo para POST /predict
        ↓
FastAPI recebe a imagem
        ↓
imagem vira tensor [1, 3, 32, 32]
        ↓
CNN faz a predição
        ↓
API retorna JSON
        ↓
frontend mostra o resultado
```

## 1. Estrutura do projeto

```text
cnn_api_fastapi_parte2/
│
├── app.py
├── model.py
├── frontend.html
├── requirements.txt
├── README.md
└── artifacts/
        ├── cifar10_cnn.pt
    └── metadata.json
```

## 2. Copiar os artefatos do notebook

Depois de executar o notebook da Parte 1, ele cria:

```text
artifacts/cifar10_cnn.pt
artifacts/metadata.json
```

Copie esses arquivos para dentro da pasta `artifacts/` deste projeto.

O resultado deve ficar assim:

```text
02_api_fastapi/artifacts/cifar10_cnn.pt
02_api_fastapi/artifacts/metadata.json
```

ou ajuste os caminhos no `app.py` para apontar para onde os arquivos estão localizados.

## 3. Criar ambiente virtual

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Instalar dependências

```bash
pip install -r requirements.txt
```

## 5. Subir a API

macOS/Linux:

```bash
python3 -m uvicorn app:app --reload
```

Windows:

```bash
python -m uvicorn app:app --reload
```

> Dica: Se a porta `8000` estiver ocupada `--port 8001` para usar outra porta, e acesse `http://127.0.0.1:8001`


## 6. Abrir o frontend

Acesse (porta padrão):

```text
http://127.0.0.1:8000/
```

Depois:

1. selecione uma imagem (`.jpg`, `.png`, etc.);
2. clique em `Classificar imagem`;
3. observe a classe prevista e o top-3;
4. veja o JSON retornado pela API.

## 7. Testar a documentação Swagger

Acesse:

```text
http://127.0.0.1:8000/docs
```

Use o endpoint:

```text
POST /predict
```

