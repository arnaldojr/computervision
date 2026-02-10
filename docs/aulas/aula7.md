# Aula 7 - Construindo uma API para Modelo de CV

## Objetivo da Aula

Criar uma API funcional que transforme um classificador de visão computacional em um serviço web, separando claramente as responsabilidades em Controller, Service e Model, elevando o nível da disciplina para integração real de modelos em sistemas.

## Conteúdo Teórico

### Arquitetura de APIs para Visão Computacional

Uma API bem projetada para visão computacional deve seguir princípios de engenharia de software:

- **Separação de Responsabilidades**: Cada componente tem uma função clara
- **Baixo Acoplamento**: Componentes devem depender minimamente uns dos outros
- **Alta Coesão**: Funcionalidades relacionadas devem estar juntas
- **Facilidade de Teste**: Componentes devem ser facilmente testáveis isoladamente

### Padrões de Arquitetura

#### MVC (Model-View-Controller)
- **Model**: Representa os dados e a lógica de negócios
- **View**: Interface com o usuário (não aplicável diretamente em APIs REST)
- **Controller**: Lida com as requisições e respostas

#### Clean Architecture
- **Entities**: Objetos de negócio centrais
- **Use Cases**: Lógica de negócios específica
- **Interface Adapters**: Adaptadores para frameworks e drivers
- **Frameworks & Drivers**: Frameworks externos e UI

### FastAPI vs Flask

**FastAPI**:
- Moderno e rápido (baseado em Starlette e Pydantic)
- Suporte nativo a tipagem e documentação automática
- Alto desempenho com async/await
- Geração automática de documentação OpenAPI

**Flask**:
- Leve e flexível
- Grande ecossistema de extensões
- Curva de aprendizado mais suave
- Menos recursos embutidos

## Atividade Prática

### Estrutura do Projeto

```
api_cv/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Ponto de entrada da API
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── vision_router.py # Rotas para visão computacional
│   │   │   └── health_router.py # Rotas de saúde do sistema
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── request_schemas.py  # Modelos de requisição
│   │       └── response_schemas.py # Modelos de resposta
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vision_service.py    # Lógica de negócio de visão computacional
│   │   └── model_service.py     # Gerenciamento de modelos
│   ├── models/
│   │   ├── __init__.py
│   │   └── vision_models.py     # Modelos de dados
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py       # Utilitários para processamento de imagem
│   │   └── validation_utils.py  # Utilitários de validação
│   └── config/
│       ├── __init__.py
│       └── settings.py          # Configurações da aplicação
├── models/                      # Diretório para modelos treinados
├── tests/                       # Testes da API
├── requirements.txt             # Dependências
└── Dockerfile                  # Para containerização
```

### Configuração Inicial

```python
# app/config/settings.py
from pydantic_settings import Settings
from typing import Optional

class Settings(Settings):
    app_name: str = "API de Visão Computacional"
    app_version: str = "1.0.0"
    debug: bool = False
    model_path: str = "models/classifier.pkl"
    allowed_image_types: list = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    api_prefix: str = "/api/v1"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Modelos de Dados (Pydantic Schemas)

```python
# app/api/schemas/request_schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class ImageClassificationRequest(BaseModel):
    """Schema para requisição de classificação de imagem"""
    image_url: Optional[str] = Field(None, description="URL da imagem para classificação")
    image_base64: Optional[str] = Field(None, description="Imagem em formato base64")

class ObjectDetectionRequest(BaseModel):
    """Schema para requisição de detecção de objetos"""
    image_url: Optional[str] = Field(None, description="URL da imagem para detecção")
    image_base64: Optional[str] = Field(None, description="Imagem em formato base64")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Limiar de confiança para detecção")

class ImageFormat(str, Enum):
    """Formatos de imagem suportados"""
    JPEG = "jpeg"
    PNG = "png"
    JPG = "jpg"
    WEBP = "webp"

class PreprocessingRequest(BaseModel):
    """Schema para requisição de pré-processamento de imagem"""
    image_url: Optional[str] = Field(None, description="URL da imagem para pré-processamento")
    image_base64: Optional[str] = Field(None, description="Imagem em formato base64")
    target_format: Optional[ImageFormat] = Field(None, description="Formato de saída desejado")
    target_size: Optional[tuple[int, int]] = Field(None, description="Tamanho de saída (largura, altura)")
    apply_grayscale: bool = Field(default=False, description="Converter para escala de cinza")
    apply_blur: bool = Field(default=False, description="Aplicar desfoque gaussiano")
    blur_kernel_size: int = Field(default=5, description="Tamanho do kernel para desfoque")
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

class ClassificationResponse(BaseResponse):
    """Resposta para classificação de imagem"""
    prediction: str
    confidence: float
    processing_time: float
    all_predictions: Optional[List[Dict[str, float]]] = None

class DetectionResponse(BaseResponse):
    """Resposta para detecção de objetos"""
    detections: List[Dict]
    count: int
    processing_time: float

class PreprocessingResponse(BaseResponse):
    """Resposta para pré-processamento de imagem"""
    processed_image_url: Optional[str] = None
    processing_time: float
    original_size: tuple[int, int]
    processed_size: tuple[int, int]

class ModelInfoResponse(BaseResponse):
    """Resposta com informações do modelo"""
    model_name: str
    model_version: str
    input_shape: tuple
    classes: List[str]
    loaded_successfully: bool
```

### Utilitários

```python
# app/utils/image_utils.py
import base64
import io
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional
from fastapi import HTTPException
import requests
from config.settings import settings

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decodifica imagem de base64 para objeto PIL"""
    try:
        # Remover cabeçalho de dados se presente
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar imagem base64: {str(e)}")

def load_image_from_url(url: str) -> Image.Image:
    """Carrega imagem de URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao carregar imagem da URL: {str(e)}")

def validate_image_content_type(content_type: str) -> bool:
    """Valida tipo de conteúdo da imagem"""
    return content_type.lower() in settings.allowed_image_types

def validate_image_size(image: Image.Image, max_pixels: int = 10000000) -> bool:
    """Valida tamanho da imagem em pixels"""
    return image.width * image.height <= max_pixels

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Converte PIL Image para numpy array"""
    return np.array(image)

def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Converte numpy array para PIL Image"""
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)

def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Redimensiona imagem mantendo proporção se necessário"""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Converte imagem para escala de cinza"""
    if image.mode != 'L':
        return image.convert('L')
    return image

def apply_gaussian_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    """Aplica desfoque gaussiano"""
    # Converter PIL para OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Aplicar desfoque
    blurred = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    
    # Converter de volta para PIL
    blurred_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    
    return blurred_pil
```

### Serviços

```python
# app/services/model_service.py
import joblib
import pickle
import os
from typing import Any, Dict
import tensorflow as tf
import numpy as np
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
                # Se não encontrar o modelo, criar um modelo mock para demonstração
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
            if model_path.endswith('.h5') or model_path.endswith('.keras'):
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
                # Tentar carregar com joblib ou pickle
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
    
    def preprocess_input(self, image_array: np.ndarray) -> np.ndarray:
        """Pré-processa entrada para o modelo"""
        # Redimensionar para o tamanho esperado pelo modelo
        if hasattr(self.model, 'input_shape'):
            expected_shape = self.model.input_shape[1:]  # Remover dimensão do batch
            if len(expected_shape) == 3:  # Imagem colorida
                from PIL import Image
                img = Image.fromarray(image_array.astype('uint8'))
                img = img.resize((expected_shape[1], expected_shape[0]))  # (width, height)
                processed = np.array(img)
            else:  # Imagem em escala de cinza
                from PIL import Image
                img = Image.fromarray(image_array.astype('uint8'))
                img = img.resize((expected_shape[1], expected_shape[0]))
                processed = np.array(img)
                if len(processed.shape) == 3:
                    processed = np.mean(processed, axis=2)  # Converter para escala de cinza
                processed = processed.reshape(processed.shape + (1,))  # Adicionar dimensão do canal
        else:
            # Para modelo mock, usar tamanho padrão
            from PIL import Image
            img = Image.fromarray(image_array.astype('uint8'))
            img = img.resize((224, 224))
            processed = np.array(img)
        
        # Normalizar valores para o intervalo [0, 1] se necessário
        if processed.max() > 1.0:
            processed = processed.astype(np.float32) / 255.0
        
        # Adicionar dimensão do batch
        if len(processed.shape) == 3:
            processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Faz predição com o modelo"""
        import time
        start_time = time.time()
        
        # Pré-processar imagem
        processed_input = self.preprocess_input(image_array)
        
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

```python
# app/services/visionservice.py
import numpy as np
from typing import Dict, List
from PIL import Image
import cv2
import time
from utils.image_utils import pil_to_numpy, numpy_to_pil
from services.model_service import ModelService

class VisionService:
    def __init__(self):
        self.model_service = ModelService()
    
    def classify_image(self, image: Image.Image) -> Dict:
        """Classifica uma imagem"""
        image_array = pil_to_numpy(image)
        return self.model_service.predict(image_array)
    
    def detect_objects(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """Detecta objetos em uma imagem (implementação mock para demonstração)"""
        start_time = time.time()
        
        # Converter imagem para array numpy
        image_array = pil_to_numpy(image)
        height, width = image_array.shape[:2]
        
        # Simular detecções (em uma implementação real, usaria um modelo como YOLO)
        # Gerar detecções mock
        mock_detections = [
            {
                "label": "pessoa",
                "confidence": 0.89,
                "bbox": [int(0.1 * width), int(0.2 * height), 
                         int(0.3 * width), int(0.4 * height)],
                "coordinates": {
                    "x_min": int(0.1 * width),
                    "y_min": int(0.2 * height),
                    "x_max": int(0.3 * width),
                    "y_max": int(0.4 * height)
                }
            },
            {
                "label": "carro",
                "confidence": 0.76,
                "bbox": [int(0.5 * width), int(0.3 * height), 
                         int(0.8 * width), int(0.7 * height)],
                "coordinates": {
                    "x_min": int(0.5 * width),
                    "y_min": int(0.3 * height),
                    "x_max": int(0.8 * width),
                    "y_max": int(0.7 * height)
                }
            }
        ]
        
        # Filtrar por threshold de confiança
        filtered_detections = [
            det for det in mock_detections 
            if det["confidence"] >= confidence_threshold
        ]
        
        processing_time = time.time() - start_time
        
        return {
            'detections': filtered_detections,
            'count': len(filtered_detections),
            'processing_time': processing_time
        }
    
    def preprocess_image(self, 
                        image: Image.Image, 
                        target_format: str = None,
                        target_size: tuple = None,
                        apply_grayscale: bool = False,
                        apply_blur: bool = False,
                        blur_kernel_size: int = 5) -> Dict:
        """Pré-processa imagem conforme especificações"""
        start_time = time.time()
        
        original_size = image.size
        
        # Aplicar transformações
        processed_image = image.copy()
        
        if apply_grayscale:
            processed_image = processed_image.convert('L').convert('RGB')  # Converter para L e depois para RGB para manter 3 canais
        
        if apply_blur:
            from utils.image_utils import apply_gaussian_blur
            processed_image = apply_gaussian_blur(processed_image, blur_kernel_size)
        
        if target_size:
            from utils.image_utils import resize_image
            processed_image = resize_image(processed_image, target_size)
        
        if target_format:
            # Ajustar formato (na verdade, o PIL já armazena internamente)
            pass
        
        processed_size = processed_image.size
        
        processing_time = time.time() - start_time
        
        return {
            'processed_image': processed_image,
            'processing_time': processing_time,
            'original_size': original_size,
            'processed_size': processed_size
        }
```

### Controladores (Routers)

```python
# app/api/routers/vision_router.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import base64
import io
from PIL import Image

from api.schemas.request_schemas import (
    ImageClassificationRequest, 
    ObjectDetectionRequest, 
    PreprocessingRequest
)
from api.schemas.response_schemas import (
    ClassificationResponse, 
    DetectionResponse, 
    PreprocessingResponse, 
    ModelInfoResponse
)
from services.vision_service import VisionService
from utils.image_utils import (
    decode_base64_image, 
    load_image_from_url, 
    validate_image_content_type,
    pil_to_numpy
)

router = APIRouter(prefix="/vision", tags=["Visão Computacional"])

# Instanciar serviço
vision_service = VisionService()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    request: ImageClassificationRequest = None,
    file: UploadFile = File(None)
):
    """
    Classifica uma imagem usando modelo de visão computacional
    """
    try:
        # Obter imagem de diferentes fontes
        image = None
        
        if file:
            # Upload via multipart form
            if not validate_image_content_type(file.content_type):
                raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        
        elif request:
            if request.image_base64:
                # Imagem em base64
                image = decode_base64_image(request.image_base64)
            elif request.image_url:
                # Imagem via URL
                image = load_image_from_url(request.image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida")
        
        # Classificar imagem
        result = vision_service.classify_image(image)
        
        return ClassificationResponse(
            success=True,
            prediction=result['prediction'],
            confidence=result['confidence'],
            processing_time=result['processing_time'],
            all_predictions=result.get('all_predictions')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/detect_objects", response_model=DetectionResponse)
async def detect_objects(
    request: ObjectDetectionRequest,
    file: UploadFile = File(None)
):
    """
    Detecta objetos em uma imagem
    """
    try:
        # Obter imagem
        image = None
        
        if file:
            if not validate_image_content_type(file.content_type):
                raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        elif request.image_base64:
            image = decode_base64_image(request.image_base64)
        elif request.image_url:
            image = load_image_from_url(request.image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida")
        
        # Detectar objetos
        result = vision_service.detect_objects(image, request.confidence_threshold)
        
        return DetectionResponse(
            success=True,
            detections=result['detections'],
            count=result['count'],
            processing_time=result['processing_time']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/preprocess", response_model=PreprocessingResponse)
async def preprocess_image(request: PreprocessingRequest):
    """
    Pré-processa uma imagem conforme especificações
    """
    try:
        # Obter imagem
        image = None
        
        if request.image_base64:
            image = decode_base64_image(request.image_base64)
        elif request.image_url:
            image = load_image_from_url(request.image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Nenhuma imagem fornecida")
        
        # Pré-processar imagem
        result = vision_service.preprocess_image(
            image=image,
            target_format=request.target_format,
            target_size=request.target_size,
            apply_grayscale=request.apply_grayscale,
            apply_blur=request.apply_blur,
            blur_kernel_size=request.blur_kernel_size
        )
        
        # Converter imagem processada para base64 para resposta (opcional)
        buffered = io.BytesIO()
        result['processed_image'].save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return PreprocessingResponse(
            success=True,
            processed_image_url=f"data:image/jpeg;base64,{img_str}",
            processing_time=result['processing_time'],
            original_size=result['original_size'],
            processed_size=result['processed_size']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Obtém informações sobre o modelo de visão computacional carregado
    """
    model_info = vision_service.model_service.get_model_info()
    
    return ModelInfoResponse(
        success=True,
        model_name=model_info['model_name'],
        model_version=model_info['model_version'],
        input_shape=model_info['input_shape'],
        classes=model_info['classes'],
        loaded_successfully=model_info['loaded_successfully']
    )
```

```python
# app/api/routers/health_router.py
from fastapi import APIRouter
from api.schemas.response_schemas import BaseResponse

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("", response_model=BaseResponse)
async def health_check():
    """
    Verifica se o serviço está ativo e saudável
    """
    return BaseResponse(
        success=True,
        message="API de Visão Computacional está ativa!",
    )

@router.get("/ready")
async def readiness_check():
    """
    Verifica se o serviço está pronto para receber requisições
    """
    # Aqui você pode adicionar verificações mais específicas
    # como conexão com banco de dados, disponibilidade de modelos, etc.
    return {"status": "ready"}
```

### Ponto de Entrada Principal

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import vision_router, health_router
from config.settings import settings

# Criar instância do FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="API para serviços de Visão Computacional"
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
app.include_router(health_router.router)
app.include_router(vision_router.router)

@app.get("/")
async def root():
    """
    Ponto de entrada raiz da API
    """
    return {
        "message": "Bem-vindo à API de Visão Computacional!",
        "version": settings.app_version,
        "documentation": "/docs",
        "redoc": "/redoc"
    }

# Para rodar: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Arquivo de Requisitos

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

### Exemplo de Teste

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
    data = response.json()
    assert data["success"] is True
    assert "message" in data

def test_classify_endpoint_with_upload():
    """Testa endpoint de classificação com upload de arquivo"""
    test_image = create_test_image()
    
    response = client.post(
        "/api/v1/vision/classify",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    assert "confidence" in data

def test_classify_endpoint_with_base64():
    """Testa endpoint de classificação com imagem em base64"""
    test_image = create_test_image()
    import base64
    
    # Converter imagem para base64
    test_image.seek(0)
    img_bytes = test_image.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    response = client.post(
        "/api/v1/vision/classify",
        json={"image_base64": f"data:image/jpeg;base64,{img_base64}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_model_info():
    """Testa endpoint de informações do modelo"""
    response = client.get("/api/v1/vision/model_info")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "model_name" in data
    assert "classes" in data
```

## Resultado Esperado

Nesta aula, você:

1. Aprendeu a arquitetura de APIs para visão computacional
2. Implementou uma API completa com FastAPI seguindo padrões de engenharia
3. Separou claramente as responsabilidades em Controller, Service e Model
4. Criou modelos de requisição e resposta com Pydantic
5. Implementou serviços para gerenciamento de modelos e processamento de imagens
6. Adicionou endpoints para classificação, detecção e pré-processamento
7. Configurou middlewares e tratamento de erros
8. Criou testes para validar a funcionalidade da API

Agora você tem uma API funcional que pode ser integrada a qualquer sistema, permitindo que outros serviços consumam suas capacidades de visão computacional.