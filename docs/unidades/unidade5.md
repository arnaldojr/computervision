# Unidade 5 - Integração e Engenharia

## Construção de APIs para Modelos de Visão Computacional

A integração de modelos de visão computacional em sistemas reais geralmente envolve a criação de APIs (Application Programming Interfaces) que permitem que outros sistemas consumam os serviços de análise visual.

### Escolha de Framework para API

Duas opções populares para criar APIs em Python são:

- **FastAPI**: Moderno, rápido e com suporte a tipagem
- **Flask**: Leve e flexível, ideal para prototipagem

### Estrutura de uma API para Visão Computacional

Uma API bem estruturada para visão computacional deve seguir princípios de engenharia de software:

```
api_vision/
├── app/
│   ├── __init__.py
│   ├── main.py              # Ponto de entrada da API
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── vision.py    # Rotas para visão computacional
│   │   │   └── health.py    # Rotas de saúde do sistema
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── request.py   # Modelos de requisição
│   │       └── response.py  # Modelos de resposta
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vision_service.py  # Lógica de negócio
│   │   └── model_loader.py    # Carregamento de modelos
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py     # Funções auxiliares para imagens
│   │   └── validation.py      # Funções de validação
│   └── config/
│       ├── __init__.py
│       └── settings.py        # Configurações do sistema
├── models/                   # Diretório para modelos treinados
├── tests/                    # Testes da API
├── requirements.txt          # Dependências
└── Dockerfile               # Para containerização
```

### Exemplo de API com FastAPI

```python
# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import cv2
from typing import Optional

# Importar serviços
from app.services.vision_service import VisionService
from app.utils.image_utils import validate_image
from app.api.models.request import ClassificationRequest
from app.api.models.response import ClassificationResponse

app = FastAPI(
    title="API de Visão Computacional",
    description="Serviço para classificação e detecção de objetos em imagens",
    version="1.0.0"
)

# Instanciar serviço de visão computacional
vision_service = VisionService()

@app.get("/")
async def root():
    return {"message": "API de Visão Computacional está ativa!"}

@app.get("/health")
async def health_check():
    """Verifica se o serviço está ativo e saudável"""
    return {"status": "healthy"}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    Classifica uma imagem usando modelo de visão computacional
    """
    try:
        # Validar tipo de arquivo
        if not validate_image(file.content_type):
            raise HTTPException(status_code=400, detail="Tipo de arquivo inválido")
        
        # Ler imagem
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Converter para formato necessário
        image_array = np.array(image)
        
        # Processar imagem e obter classificação
        result = vision_service.classify_image(image_array)
        
        return ClassificationResponse(
            success=True,
            prediction=result['prediction'],
            confidence=result['confidence'],
            processing_time=result['processing_time']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_objects")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detecta objetos em uma imagem
    """
    try:
        if not validate_image(file.content_type):
            raise HTTPException(status_code=400, detail="Tipo de arquivo inválido")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Detectar objetos
        detections = vision_service.detect_objects(image_array)
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rodar com: uvicorn app.main:app --reload
```

### Modelos de Requisição e Resposta

```python
# app/api/models/request.py
from pydantic import BaseModel
from typing import Optional

class ClassificationRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

# app/api/models/response.py
from pydantic import BaseModel
from typing import List, Dict, Any
import time

class ClassificationResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    processing_time: float
    timestamp: str = str(time.time())

class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict[str, Any]]
    count: int
    processing_time: float
```

### Serviço de Visão Computacional

```python
# app/services/vision_service.py
import time
import numpy as np
from typing import Dict, Any, List
import tensorflow as tf
from PIL import Image

class VisionService:
    def __init__(self):
        """Inicializa o serviço carregando modelos necessários"""
        self.model = self.load_model()
        self.classes = self.load_class_names()
    
    def load_model(self):
        """Carrega modelo treinado"""
        # Substituir pelo caminho real do modelo
        try:
            model = tf.keras.models.load_model('models/classifier.h5')
            return model
        except:
            # Modelo mock para demonstração
            print("Usando modelo mock - substitua pelo modelo real")
            return None
    
    def load_class_names(self):
        """Carrega nomes das classes"""
        # Substituir pela lista real de classes
        return ['classe_a', 'classe_b', 'classe_c']
    
    def preprocess_image(self, image_array):
        """Pré-processa imagem para entrada do modelo"""
        # Redimensionar imagem
        image = Image.fromarray(image_array)
        image = image.resize((224, 224))
        image_array = np.array(image)
        
        # Normalizar
        image_array = image_array.astype(np.float32) / 255.0
        
        # Expandir dimensão para batch
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def classify_image(self, image_array):
        """Classifica uma imagem"""
        start_time = time.time()
        
        # Pré-processar imagem
        processed_image = self.preprocess_image(image_array)
        
        # Fazer predição (mock se modelo não estiver carregado)
        if self.model is not None:
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.classes[predicted_class_idx]
        else:
            # Simular predição
            predicted_class = "classe_a"
            confidence = 0.85
        
        processing_time = time.time() - start_time
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'processing_time': processing_time
        }
    
    def detect_objects(self, image_array):
        """Detecta objetos em uma imagem"""
        start_time = time.time()
        
        # Converter para OpenCV
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Simular detecções (substituir por modelo real de detecção)
        height, width = image_array.shape[:2]
        detections = [
            {
                "label": "objeto",
                "confidence": 0.92,
                "bbox": [int(0.1 * width), int(0.1 * height), 
                         int(0.3 * width), int(0.3 * height)],
                "coordinates": {
                    "x_min": int(0.1 * width),
                    "y_min": int(0.1 * height),
                    "x_max": int(0.3 * width),
                    "y_max": int(0.3 * height)
                }
            }
        ]
        
        processing_time = time.time() - start_time
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'count': len(detections)
        }
```

### Utilitários

```python
# app/utils/image_utils.py
from typing import Optional

def validate_image(content_type: str) -> bool:
    """Valida se o tipo de conteúdo é uma imagem suportada"""
    valid_types = [
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/gif',
        'image/webp'
    ]
    return content_type.lower() in valid_types

def resize_image(image_array, target_size=(224, 224)):
    """Redimensiona imagem mantendo proporção"""
    from PIL import Image
    
    image = Image.fromarray(image_array)
    image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(image)

def normalize_image(image_array):
    """Normaliza imagem para valores entre 0 e 1"""
    return image_array.astype(np.float32) / 255.0
```

## Estrutura Arquitetural

### Padrões de Arquitetura

Para sistemas de visão computacional, é importante seguir padrões arquiteturais que promovam:

- **Separation of Concerns**: Separar diferentes responsabilidades
- **Scalability**: Permitir expansão horizontal
- **Maintainability**: Facilitar manutenção e evolução
- **Testability**: Permitir testes unitários e de integração

### Arquitetura em Camadas

```
┌─────────────────┐
│   Presentation  │  ← API REST, interfaces web
├─────────────────┤
│   Application   │  ← Lógica de negócio, regras de negócio
├─────────────────┤
│    Domain       │  ← Modelos de domínio, entidades
├─────────────────┤
│   Infrastructure│  ← Persistência, serviços externos
└─────────────────┘
```

### Padrão Repository

```python
# app/repositories/model_repository.py
from abc import ABC, abstractmethod
from typing import Optional
import pickle
import joblib

class ModelRepository(ABC):
    @abstractmethod
    def save_model(self, model, model_path: str):
        pass
    
    @abstractmethod
    def load_model(self, model_path: str):
        pass
    
    @abstractmethod
    def get_model_info(self, model_path: str) -> dict:
        pass

class LocalModelRepository(ModelRepository):
    def save_model(self, model, model_path: str):
        """Salva modelo localmente"""
        if hasattr(model, 'save'):
            model.save(model_path)
        else:
            joblib.dump(model, model_path)
    
    def load_model(self, model_path: str):
        """Carrega modelo localmente"""
        try:
            # Tentar carregar como Keras model primeiro
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except:
            # Caso contrário, usar joblib
            return joblib.load(model_path)
    
    def get_model_info(self, model_path: str) -> dict:
        """Obtém informações sobre o modelo"""
        import os
        stat = os.stat(model_path)
        return {
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime
        }
```

### Injeção de Dependência

```python
# app/dependencies.py
from app.services.vision_service import VisionService
from app.repositories.model_repository import LocalModelRepository

def get_vision_service():
    """Dependency injection para VisionService"""
    model_repo = LocalModelRepository()
    # Aqui poderia injetar o repositório no serviço
    return VisionService()

# Em rotas:
# vision_service = Depends(get_vision_service)
```

## Containerização

### Dockerfile para API de Visão Computacional

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar dependências
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretório para modelos
RUN mkdir -p models

# Expor porta
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Arquivo de Requisitos

```txt
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
tensorflow==2.15.0
opencv-python==4.8.1.78
pillow==10.1.0
numpy==1.24.3
pydantic==2.5.0
joblib==1.3.2
python-multipart==0.0.6
```

### Docker Compose para Ambiente Completo

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/models/classifier.h5
      - LOG_LEVEL=info
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  redis_data:
```

### Configuração de Nginx para Proxy Reverso

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_server {
        server api:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://api_server;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Configurações para uploads grandes
            client_max_body_size 10M;
        }
        
        # Health check
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

## Boas Práticas e Organização

### Versionamento de Modelos

É crucial manter controle de versão dos modelos de ML:

```python
# app/utils/model_versioning.py
import os
import json
from datetime import datetime
from typing import Dict, Any

class ModelVersionManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.version_file = os.path.join(models_dir, "versions.json")
    
    def register_model(self, model_name: str, model_path: str, metadata: Dict[str, Any]):
        """Registra novo modelo com metadados"""
        versions = self._load_versions()
        
        version_info = {
            "path": model_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "active": True
        }
        
        if model_name not in versions:
            versions[model_name] = {}
        
        # Criar nova versão
        version_id = f"v{len(versions[model_name]) + 1:03d}"
        versions[model_name][version_id] = version_info
        
        # Desativar versões anteriores
        for version_key in versions[model_name]:
            if version_key != version_id:
                versions[model_name][version_key]["active"] = False
        
        self._save_versions(versions)
        return version_id
    
    def get_active_model(self, model_name: str) -> Dict[str, Any]:
        """Obtém o modelo ativo"""
        versions = self._load_versions()
        
        if model_name not in versions:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        for version_id, info in versions[model_name].items():
            if info["active"]:
                return {**info, "version": version_id, "name": model_name}
        
        raise ValueError(f"Nenhuma versão ativa encontrada para {model_name}")
    
    def _load_versions(self) -> Dict[str, Any]:
        """Carrega arquivo de versões"""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self, versions: Dict[str, Any]):
        """Salva arquivo de versões"""
        os.makedirs(self.models_dir, exist_ok=True)
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)

# Exemplo de uso
# manager = ModelVersionManager()
# version = manager.register_model(
#     "classifier", 
#     "models/classifier_v001.h5",
#     {"accuracy": 0.92, "dataset": "custom_dataset_v2"}
# )
```

### Logging e Monitoramento

```python
# app/utils/logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(log_level="INFO", log_file=None):
    """Configura logging para a aplicação"""
    logger = logging.getLogger("vision_api")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Formatação
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (opcional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Uso no serviço
logger = setup_logging(log_level="INFO", log_file="logs/api.log")

def log_prediction(image_path: str, prediction: str, confidence: float):
    """Registra predição para auditoria"""
    logger.info(f"Prediction: image={image_path}, prediction={prediction}, confidence={confidence:.3f}")
```

### Testes Automatizados

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np

from app.main import app

client = TestClient(app)

def create_test_image(width=224, height=224):
    """Cria imagem de teste"""
    image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def test_health_endpoint():
    """Testa endpoint de saúde"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_classify_endpoint():
    """Testa endpoint de classificação"""
    test_image = create_test_image()
    
    response = client.post(
        "/classify",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_invalid_image_type():
    """Testa upload de tipo de imagem inválido"""
    # Criar conteúdo inválido
    invalid_content = io.BytesIO(b"invalid image content")
    
    response = client.post(
        "/classify",
        files={"file": ("test.txt", invalid_content, "text/plain")}
    )
    
    assert response.status_code == 400

# tests/test_vision_service.py
from unittest.mock import Mock, patch
import numpy as np

from app.services.vision_service import VisionService

class TestVisionService:
    def test_preprocess_image(self):
        """Testa pré-processamento de imagem"""
        service = VisionService()
        
        # Criar imagem de teste
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        processed = service.preprocess_image(test_image)
        
        # Verificar formato
        assert processed.shape == (1, 224, 224, 3)  # Batch, height, width, channels
        assert processed.dtype == np.float32
        assert processed.max() <= 1.0  # Normalizado
        assert processed.min() >= 0.0  # Normalizado
    
    @patch('tensorflow.keras.models.load_model')
    def test_classify_image(self, mock_load_model):
        """Testa classificação de imagem"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # Probabilidades
        mock_load_model.return_value = mock_model
        
        service = VisionService()
        service.model = mock_model
        service.classes = ['classe_a', 'classe_b', 'classe_c']
        
        # Criar imagem de teste
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result = service.classify_image(test_image)
        
        # Verificar resultados
        assert result['prediction'] == 'classe_b'  # Maior probabilidade
        assert result['confidence'] == 0.8
        assert 'processing_time' in result
        assert isinstance(result['processing_time'], float)
```

### Documentação da API

A API criada com FastAPI automaticamente gera documentação interativa em `/docs` e `/redoc`.

Adicionalmente, é importante documentar:

- **Endpoints**: Quais endpoints estão disponíveis e como usá-los
- **Parâmetros**: Quais parâmetros são esperados
- **Exemplos de uso**: Como consumir a API
- **Erros comuns**: Possíveis erros e como tratá-los

### Segurança

```python
# app/security.py
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer
import os

security = HTTPBearer()

def verify_api_key(credentials):
    """Verifica chave de API"""
    expected_key = os.getenv("API_KEY", "dev-key")
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave de API inválida"
        )

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Implementa rate limiting (simplificado)"""
    # Em produção, usar Redis ou outro sistema de cache
    pass
```

## Considerações Finais

A integração de modelos de visão computacional em sistemas reais requer atenção a diversos aspectos além do modelo em si:

1. **Desempenho**: Otimizar tempo de inferência e uso de memória
2. **Escalabilidade**: Preparar o sistema para lidar com aumento de carga
3. **Monitoramento**: Acompanhar métricas de desempenho e utilização
4. **Manutenção**: Facilitar atualizações e rollback de modelos
5. **Segurança**: Proteger endpoints e dados sensíveis
6. **Observabilidade**: Registrar logs e métricas para troubleshooting

Esses elementos são tão importantes quanto o modelo de IA em si para garantir que a solução seja sustentável e eficaz em ambiente de produção.