# CP3 - Projeto Final

## Objetivo

Desenvolver um sistema completo de visão computacional que demonstre todas as habilidades aprendidas ao longo da disciplina, integrando pipeline de processamento, classificação, detecção de objetos e disponibilização em uma API.

## Opções de Projeto

Os alunos podem escolher entre as seguintes opções:

1. **Detector de EPI** - Sistema para identificação de equipamentos de proteção individual
2. **Classificador de Defeitos Industriais** - Sistema para detecção de falhas em produtos
3. **Sistema de Contagem** - Contador automático de objetos em imagens
4. **Detector de Anomalias** - Sistema para identificação de padrões anômalos
5. **OCR Simples** - Sistema para reconhecimento óptico de caracteres
6. **Reconhecimento Facial Básico** - Sistema para identificação de faces

## Requisitos Mínimos

Seu projeto deve contemplar:

### Arquitetura (30%)
- Código estruturado seguindo boas práticas de engenharia de software
- Separação clara de responsabilidades (modelos, serviços, utilitários)
- Documentação adequada do código e da arquitetura

### Funcionalidade (30%)
- Sistema completo funcionando
- Pipeline de visão computacional implementado
- Integração com modelo de ML/DL
- API para consumo do sistema

### Organização (20%)
- Estrutura de projeto bem organizada
- Código limpo e legível
- Seguimento de convenções de nomenclatura

### Clareza Técnica (20%)
- Apresentação técnica do projeto
- Explicação clara das técnicas utilizadas
- Justificativa das escolhas técnicas

## Estrutura do Projeto

```
projeto_final/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── cv_router.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── request_schemas.py
│   │       └── response_schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cv_service.py
│   │   └── model_service.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cv_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── preprocessing.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── models/
│   └── trained_model.h5 (ou .pkl)
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── tests/
├── docs/
├── requirements.txt
├── README.md
├── Dockerfile
└── docker-compose.yml
```

## Implementação Detalhada

### 1. Configuração do Projeto

```python
# app/config/settings.py
from pydantic_settings import Settings
from typing import List

class Settings(Settings):
    app_name: str = "Projeto Final - Visão Computacional"
    app_version: str = "1.0.0"
    debug: bool = False
    model_path: str = "models/trained_model.h5"
    allowed_image_types: List[str] = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    api_prefix: str = "/api/v1"
    project_type: str = "epi_detector"  # Pode ser alterado conforme o projeto

settings = Settings()
```

### 2. Modelos de Requisição e Resposta

```python
# app/api/schemas/request_schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class CVRequest(BaseModel):
    """Schema base para requisições de visão computacional"""
    image_url: Optional[str] = Field(None, description="URL da imagem")
    image_base64: Optional[str] = Field(None, description="Imagem em base64")

class EpiDetectionRequest(CVRequest):
    """Schema para requisição de detecção de EPI"""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class DefectClassificationRequest(CVRequest):
    """Schema para requisição de classificação de defeitos"""
    pass

class CountingRequest(CVRequest):
    """Schema para requisição de contagem"""
    pass
```

```python
# app/api/schemas/response_schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class BaseResponse(BaseModel):
    """Resposta base para todas as operações"""
    success: bool
    message: Optional[str] = None
    timestamp: datetime = datetime.now()

class EpiDetectionResponse(BaseResponse):
    """Resposta para detecção de EPI"""
    detections: List[Dict]
    has_protective_equipment: bool
    equipment_types: List[str]
    processing_time: float

class DefectClassificationResponse(BaseResponse):
    """Resposta para classificação de defeitos"""
    defect_type: str
    confidence: float
    severity: str
    processing_time: float

class CountingResponse(BaseResponse):
    """Resposta para contagem"""
    count: int
    objects: List[Dict]
    processing_time: float
```

### 3. Serviços de Visão Computacional

```python
# app/services/cv_service.py
import numpy as np
from PIL import Image
import time
from typing import Dict, List
from utils.image_utils import pil_to_numpy
from services.model_service import ModelService

class CVService:
    def __init__(self):
        self.model_service = ModelService()
    
    def detect_epi(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """Detecta EPI em imagem"""
        start_time = time.time()
        
        # Converter imagem para array
        image_array = pil_to_numpy(image)
        
        # Em uma implementação real, usaria um modelo treinado para detecção de EPI
        # Aqui está uma simulação
        height, width = image_array.shape[:2]
        
        # Simular detecções de EPI
        mock_detections = [
            {
                "type": "capacete",
                "confidence": 0.89,
                "bbox": [int(0.4*width), int(0.1*height), int(0.6*width), int(0.3*height)],
                "coordinates": {
                    "x_min": int(0.4*width),
                    "y_min": int(0.1*height),
                    "x_max": int(0.6*width),
                    "y_max": int(0.3*height)
                }
            },
            {
                "type": "colete",
                "confidence": 0.76,
                "bbox": [int(0.3*width), int(0.3*height), int(0.7*width), int(0.7*height)],
                "coordinates": {
                    "x_min": int(0.3*width),
                    "y_min": int(0.3*height),
                    "x_max": int(0.7*width),
                    "y_max": int(0.7*height)
                }
            }
        ]
        
        # Filtrar por threshold
        filtered_detections = [
            det for det in mock_detections 
            if det["confidence"] >= confidence_threshold
        ]
        
        has_protective_equipment = len(filtered_detections) > 0
        equipment_types = list(set([det["type"] for det in filtered_detections]))
        
        processing_time = time.time() - start_time
        
        return {
            'detections': filtered_detections,
            'has_protective_equipment': has_protective_equipment,
            'equipment_types': equipment_types,
            'processing_time': processing_time
        }
    
    def classify_defect(self, image: Image.Image) -> Dict:
        """Classifica defeito industrial"""
        start_time = time.time()
        
        # Converter imagem para array
        image_array = pil_to_numpy(image)
        
        # Simular classificação de defeito
        # Em uma implementação real, usaria modelo treinado
        defect_types = ["trinca", "falta_material", "deformacao", "sujeira", "nenhum"]
        defect_probs = np.random.dirichlet(np.ones(len(defect_types)))  # Distribuição aleatória
        
        max_idx = np.argmax(defect_probs)
        defect_type = defect_types[max_idx]
        confidence = float(defect_probs[max_idx])
        
        # Determinar severidade com base no tipo e confiança
        severity = "baixa" if confidence < 0.5 else "media" if confidence < 0.8 else "alta"
        
        processing_time = time.time() - start_time
        
        return {
            'defect_type': defect_type,
            'confidence': confidence,
            'severity': severity,
            'processing_time': processing_time
        }
    
    def count_objects(self, image: Image.Image) -> Dict:
        """Conta objetos em imagem"""
        start_time = time.time()
        
        # Converter imagem para array
        image_array = pil_to_numpy(image)
        
        # Simular contagem de objetos
        # Em uma implementação real, usaria detecção de objetos ou segmentação
        height, width = image_array.shape[:2]
        
        # Simular detecção de objetos
        num_objects = np.random.randint(1, 10)  # Contagem aleatória para simulação
        
        # Gerar bounding boxes simuladas
        objects = []
        for i in range(num_objects):
            x = np.random.randint(0, width//2)
            y = np.random.randint(0, height//2)
            w = np.random.randint(width//8, width//4)
            h = np.random.randint(height//8, height//4)
            
            objects.append({
                "id": i+1,
                "bbox": [x, y, x+w, y+h],
                "coordinates": {
                    "x_min": x,
                    "y_min": y,
                    "x_max": x+w,
                    "y_max": y+h
                }
            })
        
        processing_time = time.time() - start_time
        
        return {
            'count': num_objects,
            'objects': objects,
            'processing_time': processing_time
        }
```

### 4. Serviço de Modelos

```python
# app/services/model_service.py
import tensorflow as tf
import joblib
import pickle
import os
import numpy as np
from typing import Any, Dict
from config.settings import settings

class ModelService:
    def __init__(self):
        self.model = None
        self.model_info = {}
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo treinado"""
        try:
            model_path = settings.model_path
            
            if not os.path.exists(model_path):
                print(f"Aviso: Modelo não encontrado em {model_path}. Criando modelo mock.")
                self.model = self._create_mock_model()
                self.model_info = {
                    'model_name': 'Mock Model',
                    'model_version': '1.0.0',
                    'input_shape': (224, 224, 3),
                    'classes': ['classe_a', 'classe_b', 'classe_c'],
                    'loaded_successfully': False
                }
                return
            
            # Tentar carregar como modelo TensorFlow/Keras primeiro
            if model_path.endswith(('.h5', '.keras')):
                self.model = tf.keras.models.load_model(model_path)
                # Extrair informações do modelo
                self.model_info = {
                    'model_name': os.path.basename(model_path),
                    'model_version': '1.0.0',
                    'input_shape': self.model.input_shape,
                    'classes': getattr(self.model, 'classes', ['classe_a', 'classe_b', 'classe_c']),
                    'loaded_successfully': True
                }
            else:
                # Carregar com joblib ou pickle
                try:
                    self.model = joblib.load(model_path)
                except:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                
                # Informações para modelos sklearn
                self.model_info = {
                    'model_name': os.path.basename(model_path),
                    'model_version': '1.0.0',
                    'input_shape': 'Variável (depende do modelo)',
                    'classes': list(getattr(self.model, 'classes_', ['classe_a', 'classe_b', 'classe_c'])),
                    'loaded_successfully': True
                }
                
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            # Criar modelo mock
            self.model = self._create_mock_model()
            self.model_info = {
                'model_name': 'Mock Model',
                'model_version': '1.0.0',
                'input_shape': (224, 224, 3),
                'classes': ['classe_a', 'classe_b', 'classe_c'],
                'loaded_successfully': False
            }
    
    def _create_mock_model(self):
        """Cria modelo mock para demonstração"""
        class MockModel:
            def predict(self, X):
                # Simular predições
                n_samples = X.shape[0]
                return np.random.rand(n_samples, 3)  # 3 classes
                
            def predict_proba(self, X):
                # Simular probabilidades
                n_samples = X.shape[0]
                probs = np.random.rand(n_samples, 3)
                probs = probs / probs.sum(axis=1, keepdims=True)  # Normalizar
                return probs
        
        return MockModel()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado"""
        return self.model_info
    
    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Faz predição com o modelo"""
        import time
        start_time = time.time()
        
        # Redimensionar imagem para o tamanho esperado pelo modelo
        from PIL import Image
        img = Image.fromarray(image_array.astype('uint8'))
        img = img.resize((224, 224))
        processed_input = np.array(img).astype(np.float32) / 255.0
        processed_input = np.expand_dims(processed_input, axis=0)
        
        # Fazer predição
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_input)[0]
        else:
            predictions = self.model.predict(processed_input)[0]
            if predictions.ndim == 0:  # Se for um escalar
                probabilities = np.zeros(len(self.model_info['classes']))
                probabilities[int(predictions)] = 1.0
            else:
                probabilities = predictions
        
        # Obter classe com maior probabilidade
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.model_info['classes'][predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Obter todas as predições com probabilidades
        all_predictions = [
            {"class": cls, "probability": float(prob)}
            for cls, prob in zip(self.model_info['classes'], probabilities)
        ]
        
        processing_time = time.time() - start_time
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'processing_time': processing_time
        }
```

### 5. Router de Visão Computacional

```python
# app/api/routers/cv_router.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import base64
import io
from PIL import Image

from api.schemas.request_schemas import (
    EpiDetectionRequest, 
    DefectClassificationRequest, 
    CountingRequest
)
from api.schemas.response_schemas import (
    EpiDetectionResponse, 
    DefectClassificationResponse, 
    CountingResponse
)
from services.cv_service import CVService
from utils.image_utils import (
    decode_base64_image, 
    load_image_from_url, 
    validate_image_content_type
)

router = APIRouter(prefix="/cv", tags=["Visão Computacional"])

# Instanciar serviço
cv_service = CVService()

@router.post("/detect-epi", response_model=EpiDetectionResponse)
async def detect_epi(
    request: EpiDetectionRequest = None,
    file: UploadFile = File(None)
):
    """
    Detecta equipamentos de proteção individual em imagem
    """
    try:
        # Obter imagem
        image = None
        
        if file:
            if not validate_image_content_type(file.content_type):
                raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        elif request:
            if request.image_base64:
                image = decode_base64_image(request.image_base64)
            elif request.image_url:
                image = load_image_from_url(request.image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida")
        
        # Detectar EPI
        result = cv_service.detect_epi(image, request.confidence_threshold)
        
        return EpiDetectionResponse(
            success=True,
            detections=result['detections'],
            has_protective_equipment=result['has_protective_equipment'],
            equipment_types=result['equipment_types'],
            processing_time=result['processing_time']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/classify-defect", response_model=DefectClassificationResponse)
async def classify_defect(
    request: DefectClassificationRequest = None,
    file: UploadFile = File(None)
):
    """
    Classifica defeitos industriais em imagem
    """
    try:
        # Obter imagem
        image = None
        
        if file:
            if not validate_image_content_type(file.content_type):
                raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        elif request:
            if request.image_base64:
                image = decode_base64_image(request.image_base64)
            elif request.image_url:
                image = load_image_from_url(request.image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida")
        
        # Classificar defeito
        result = cv_service.classify_defect(image)
        
        return DefectClassificationResponse(
            success=True,
            defect_type=result['defect_type'],
            confidence=result['confidence'],
            severity=result['severity'],
            processing_time=result['processing_time']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/count-objects", response_model=CountingResponse)
async def count_objects(
    request: CountingRequest = None,
    file: UploadFile = File(None)
):
    """
    Conta objetos em imagem
    """
    try:
        # Obter imagem
        image = None
        
        if file:
            if not validate_image_content_type(file.content_type):
                raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        elif request:
            if request.image_base64:
                image = decode_base64_image(request.image_base64)
            elif request.image_url:
                image = load_image_from_url(request.image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida")
        
        # Contar objetos
        result = cv_service.count_objects(image)
        
        return CountingResponse(
            success=True,
            count=result['count'],
            objects=result['objects'],
            processing_time=result['processing_time']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """
    Retorna informações sobre o modelo carregado
    """
    model_info = cv_service.model_service.get_model_info()
    return {
        "success": True,
        "model_info": model_info
    }
```

### 6. Arquivo Principal

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.cv_router import router as cv_router
from config.settings import settings

# Criar instância do FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="Projeto Final - Sistema Completo de Visão Computacional"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substituir por domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir roteadores
app.include_router(cv_router)

@app.get("/")
async def root():
    """
    Ponto de entrada raiz da API
    """
    return {
        "message": "Projeto Final - Visão Computacional",
        "version": settings.app_version,
        "documentation": "/docs",
        "available_routes": [
            "/api/v1/cv/detect-epi",
            "/api/v1/cv/classify-defect",
            "/api/v1/cv/count-objects"
        ]
    }

@app.get("/health")
async def health():
    """
    Endpoint de saúde do sistema
    """
    return {"status": "healthy", "service": "cv-project-api"}

# Para rodar: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Arquivo de Requisitos

```txt
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
Pillow==10.1.0
numpy==1.24.3
opencv-python==4.8.1.78
scikit-learn==1.3.0
tensorflow==2.15.0
joblib==1.3.2
requests==2.31.0
python-multipart==0.0.6
python-dotenv==1.0.0
```

### 8. Documentação do README

```markdown
# Projeto Final - Visão Computacional Aplicada

Este projeto implementa um sistema completo de visão computacional com múltiplas funcionalidades.

## Funcionalidades

- Detecção de Equipamentos de Proteção Individual (EPI)
- Classificação de Defeitos Industriais
- Contagem de Objetos em Imagens

## Tecnologias Utilizadas

- FastAPI: Framework web moderno e rápido
- TensorFlow: Modelos de Deep Learning
- OpenCV: Processamento de imagens
- Docker: Containerização
- Pydantic: Validação de dados

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Coloque seu modelo treinado em `models/trained_model.h5`
5. Atualize as configurações em `app/config/settings.py`

## Execução

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /` - Página inicial
- `GET /health` - Status de saúde da API
- `POST /api/v1/cv/detect-epi` - Detecção de EPI
- `POST /api/v1/cv/classify-defect` - Classificação de defeitos
- `POST /api/v1/cv/count-objects` - Contagem de objetos
- `GET /api/v1/cv/model-info` - Informações sobre o modelo

## Documentação Automática

Acesse `/docs` ou `/redoc` para documentação interativa da API.

## Avaliação

O projeto será avaliado com base nos seguintes critérios:

- **Arquitetura (30%)**: Código estruturado e seguindo boas práticas
- **Funcionalidade (30%)**: Sistema completo funcionando
- **Organização (20%)**: Código limpo e bem organizado
- **Clareza Técnica (20%)**: Apresentação e explicação clara das técnicas

## Autores

[Seu nome]
[Seu email]
```

## Critérios de Avaliação

### Arquitetura (30%)
- Organização do código em módulos bem definidos
- Separação de responsabilidades (models, services, utils, etc.)
- Seguimento de padrões de projeto
- Documentação do código

### Funcionamento (30%)
- Sistema completo e funcional
- Integração adequada entre componentes
- API respondendo corretamente às requisições
- Modelos de ML/DL integrados corretamente

### Organização (20%)
- Estrutura de diretórios clara e lógica
- Código limpo e legível
- Convenções de nomenclatura consistentes
- Arquivos de configuração adequados

### Clareza Técnica (20%)
- Apresentação clara do projeto
- Explicação das técnicas utilizadas
- Justificativa das escolhas técnicas
- Demonstração de compreensão dos conceitos

## Entrega

A entrega deve incluir:

1. Código-fonte completo e funcional
2. Documentação adequada
3. README com instruções de execução
4. Apresentação técnica do projeto
5. Demonstração do sistema em funcionamento