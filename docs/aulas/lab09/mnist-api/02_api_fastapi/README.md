# Parte 2 — PyTorch + FastAPI + Frontend com Canvas

Vamos usar o modelo treinado no notebook como motor de uma API.

## Fluxo da aplicação

```text
aluno desenha no navegador
        ↓
canvas gera uma imagem PNG
        ↓
fetch envia para POST /predict
        ↓
FastAPI recebe a imagem
        ↓
imagem vira tensor [1, 1, 28, 28]
        ↓
modelo PyTorch faz a predição
        ↓
API retorna JSON
        ↓
frontend mostra o resultado
```

## 1. Estrutura do projeto

```text
mnist_api_fastapi_parte2_com_frontend/
│
├── app.py
├── model.py
├── frontend.html
├── requirements.txt
├── README.md
└── artifacts/
    ├── mnist_mlp.pt
    └── metadata.json
```

## 2. Copiar os artefatos do notebook

Depois de executar o notebook da Parte 1, ele cria:

```text
artifacts/mnist_mlp.pt
artifacts/metadata.json
```

Copie esses arquivos para dentro da pasta `artifacts/` deste projeto.

O resultado deve ficar assim:

```text
02_api_fastapi/artifacts/mnist_mlp.pt
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

Acesse:

```text
http://127.0.0.1:8000/
```

Depois:

1. desenhe um dígito de 0 a 9;
2. clique em `Prever dígito`;
3. observe a classe prevista;
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

