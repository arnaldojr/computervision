# Aula 10 - Detecção Moderna (YOLO)

## Objetivo da Aula

Implementar detecção de objetos moderna com YOLO (You Only Look Once), entender bounding boxes, tempo de inferência, quantização e containerização de modelos.

## Conteúdo Teórico

### Detecção de Objetos Moderna

A detecção de objetos é uma tarefa fundamental em visão computacional que combina classificação e localização. Diferente da classificação de imagens, a detecção de objetos identifica:

- **O que** está na imagem (classificação)
- **Onde** está na imagem (localização com bounding boxes)
- **Quão certo** está da detecção (confiança)

### YOLO (You Only Look Once)

YOLO é uma arquitetura popular para detecção de objetos em tempo real que processa a imagem inteira em uma única passagem:

**Vantagens:**
- Alta velocidade de inferência
- Boa acurácia para aplicações em tempo real
- Arquitetura elegante e eficiente

**Componentes principais:**
- Grid de detecção
- Bounding boxes com confiança
- Classificação de objetos
- Non-maximum suppression

### Bounding Boxes

As bounding boxes são retângulos que delimitam objetos detectados, tipicamente representados por:
- **(x, y, w, h)**: Coordenadas do centro e dimensões
- **(x_min, y_min, x_max, y_max)**: Coordenadas dos cantos

### Aplicações Reais

- Segurança: Detecção de pessoas e veículos
- Varejo: Análise de comportamento do cliente
- Agricultura: Detecção de pragas e colheitas
- Saúde: Identificação de anomalias em imagens médicas

## Atividade Prática

### Implementar Detector de Objetos

```python
# src/models/yolo_detector.py
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
import time

class YOLODetector:
    def __init__(self, model_path: str = None, config_path: str = None, classes_path: str = None):
        """
        Inicializa o detector YOLO
        Nota: Esta é uma implementação conceitual
        Para uso real, recomenda-se ultralytics/yolov5 ou tensorflow-models
        """
        self.model_path = model_path
        self.config_path = config_path
        self.classes_path = classes_path
        self.net = None
        self.classes = []
        self.output_layers = []
        
        # Em uma implementação real, carregaríamos o modelo aqui
        self._setup_mock_model()
    
    def _setup_mock_model(self):
        """Configura modelo mock para demonstração"""
        print("Configurando modelo mock para demonstração...")
        
        # Classes comuns para demonstração
        self.classes = [
            'pessoa', 'bicicleta', 'carro', 'moto', 'avião', 'ônibus', 
            'trem', 'caminhão', 'barco', 'semáforo', 'hidrante', 
            'placa de parar', 'estacionamento', 'banco', 'árvore', 
            'luz vermelha', 'sinal de mão', 'cone', 'pássaro', 'gato', 
            'cachorro', 'cavalo', 'ovelha', 'vaca', 'elefante', 
            'urso', 'zebra', 'girafa', 'mochila', 'guarda-chuva', 
            'bolsa', 'gravata', 'maleta', 'frisbee', 'skate', 
            'surf', 'bola', 'pipa', 'bastão', 'luva', 'tênis', 
            'shorts', 'camisa', 'vestido', 'casaco', 'meia', 
            'óculos', 'relógio', 'bolsa', 'chave', 'carteira', 
            'celular', 'perfume', 'espelho', 'escova', 'batom', 
            'livro', 'caneta', 'lápis', 'caderno', 'borracha', 
            'cola', 'tesoura', 'grampeador', 'clipe', 'agulha', 
            'linha', 'botão', 'fita', 'tesoura', 'martelo', 
            'prego', 'parafuso', 'arruela', 'porca', 'chave', 
            'faca', 'garfo', 'colher', 'copo', 'prato', 'panela', 
            'frigideira', 'fogão', 'geladeira', 'microondas', 
            'forno', 'liquidificador', 'cafeteira', 'torradeira', 
            'batedeira', 'cortador', 'ralador', 'abridor', 
            'faca', 'tábua', 'pia', 'fogão', 'geladeira'
        ]
        
        # Limitar para as primeiras 10 classes para simplificar
        self.classes = self.classes[:10]
    
    def load_model(self):
        """Carrega modelo YOLO (implementação real)"""
        if self.model_path and self.config_path:
            self.net = cv2.dnn.readNet(self.model_path, self.config_path)
            
            # Obter nomes das camadas de saída
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        else:
            print("Usando modelo mock para demonstração")
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detecta objetos na imagem
        Retorna lista de dicionários com informações sobre detecções
        """
        start_time = time.time()
        
        height, width, channels = image.shape
        
        # Em uma implementação real, faríamos:
        # 1. Criar blob da imagem
        # 2. Passar pela rede
        # 3. Processar outputs
        
        # Para esta demonstração, simularemos detecções
        detections = self._simulate_detections(image, confidence_threshold)
        
        processing_time = time.time() - start_time
        
        # Adicionar tempo de processamento às detecções
        for detection in detections:
            detection['processing_time'] = processing_time / len(detections) if detections else processing_time
        
        return detections
    
    def _simulate_detections(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """Simula detecções para demonstração"""
        height, width = image.shape[:2]
        
        # Gerar detecções simuladas
        num_detections = np.random.randint(1, 5)  # 1 a 4 detecções
        detections = []
        
        for _ in range(num_detections):
            # Gerar bounding box aleatória
            x = np.random.randint(0, width // 2)
            y = np.random.randint(0, height // 2)
            w = np.random.randint(width // 4, width // 2)
            h = np.random.randint(height // 4, height // 2)
            
            # Garantir que a bounding box esteja dentro dos limites
            x = min(x, width - w)
            y = min(y, height - h)
            
            # Classe aleatória
            class_id = np.random.randint(0, len(self.classes))
            class_name = self.classes[class_id]
            
            # Confiança aleatória acima do threshold
            confidence = np.random.uniform(confidence_threshold, 1.0)
            
            detection = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [int(x), int(y), int(w), int(h)],
                'coordinates': {
                    'x_min': int(x),
                    'y_min': int(y),
                    'x_max': int(x + w),
                    'y_max': int(y + h)
                }
            }
            
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True) -> np.ndarray:
        """Desenha detecções na imagem"""
        result_image = image.copy()
        
        # Cores para desenho (BGR)
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        for detection in detections:
            # Obter coordenadas
            x, y, w, h = detection['bbox']
            
            # Obter cor para esta classe
            class_id = detection['class_id']
            color = [int(c) for c in colors[class_id]]
            
            # Desenhar bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Preparar texto
            label = detection['class_name']
            if show_confidence:
                label += f" {detection['confidence']:.2f}"
            
            # Calcular tamanho do texto
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Desenhar fundo para o texto
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Colocar texto
            cv2.putText(result_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo"""
        return {
            'model_type': 'YOLO (simulado para demonstração)',
            'classes_count': len(self.classes),
            'classes': self.classes,
            'input_resolution': 'Variável',
            'real_model_available': self.model_path is not None
        }
```

### Implementar Quantização e Otimização

```python
# src/models/optimization.py
import tensorflow as tf
import numpy as np
from typing import Any

class ModelOptimizer:
    def __init__(self, model):
        self.model = model
    
    def quantize_model(self) -> bytes:
        """Converte modelo para versão quantizada (TensorFlow Lite)"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Configurar quantização
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Converter modelo
        quantized_model = converter.convert()
        
        return quantized_model
    
    def quantize_with_calibration(self, calibration_dataset) -> bytes:
        """Quantiza modelo com calibração"""
        def representative_dataset():
            for _ in range(100):  # Usar 100 amostras para calibração
                data = next(calibration_dataset)
                yield [data]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        # Configurar inferência de int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        quantized_model = converter.convert()
        
        return quantized_model
    
    def benchmark_model(self, tflite_model: bytes, test_data: np.ndarray) -> Dict[str, Any]:
        """Avalia desempenho do modelo TensorFlow Lite"""
        import time
        
        # Carregar modelo TFLite
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Obter detalhes de entrada e saída
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Preparar dados de teste
        input_shape = input_details[0]['shape']
        
        # Medir tempo de inferência
        inference_times = []
        
        for i in range(len(test_data)):
            # Preparar entrada
            input_data = test_data[i:i+1].astype(np.float32)
            
            # Definir tensor de entrada
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Medir tempo de inferência
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
        
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'model_size_mb': len(tflite_model) / (1024 * 1024),
            'num_inferences': len(test_data)
        }
    
    def compare_models(self, original_model, quantized_model_bytes, test_data: np.ndarray) -> Dict[str, Any]:
        """Compara desempenho de modelo original e quantizado"""
        # Testar modelo original
        original_start = time.time()
        original_predictions = original_model.predict(test_data[:10])  # Testar com poucas amostras
        original_time = time.time() - original_start
        
        # Testar modelo quantizado
        interpreter = tf.lite.Interpreter(model_content=quantized_model_bytes)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        quantized_times = []
        
        for i in range(min(10, len(test_data))):
            input_data = test_data[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            start = time.time()
            interpreter.invoke()
            quantized_times.append(time.time() - start)
        
        avg_quantized_time = np.mean(quantized_times)
        
        return {
            'original_time': original_time,
            'quantized_time': avg_quantized_time,
            'speedup': original_time / avg_quantized_time if avg_quantized_time > 0 else 0,
            'original_size_mb': original_model.count_params() * 4 / (1024 * 1024),  # Aproximado
            'quantized_size_mb': len(quantized_model_bytes) / (1024 * 1024)
        }
```

### Implementar Containerização

```python
# src/deployment/containerizer.py
import docker
import os
from typing import Dict, List
import tempfile

class ModelContainerizer:
    def __init__(self):
        self.client = docker.from_env()
    
    def create_dockerfile_content(self, model_path: str, requirements: List[str] = None) -> str:
        """Cria conteúdo do Dockerfile para modelo"""
        if requirements is None:
            requirements = [
                'tensorflow==2.15.0',
                'opencv-python==4.8.1.78',
                'numpy==1.24.3',
                'Pillow==10.1.0',
                'fastapi==0.104.1',
                'uvicorn[standard]==0.24.0'
            ]
        
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar modelo
COPY {model_path} /app/model/

# Copiar código da aplicação
COPY . /app/

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expor porta
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        return dockerfile_content
    
    def create_requirements_txt(self, requirements: List[str]) -> str:
        """Cria conteúdo do requirements.txt"""
        return '\n'.join(requirements)
    
    def build_container(self, dockerfile_content: str, requirements_content: str, 
                       model_path: str, tag: str, build_context: str = ".") -> Dict[str, any]:
        """Constroi container Docker"""
        # Criar arquivos temporários
        with tempfile.TemporaryDirectory() as temp_dir:
            # Salvar Dockerfile
            dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Salvar requirements.txt
            requirements_path = os.path.join(temp_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            
            try:
                # Fazer build da imagem
                image, build_logs = self.client.images.build(
                    path=temp_dir,
                    tag=tag,
                    rm=True,
                    dockerfile='Dockerfile'
                )
                
                # Coletar logs de build
                build_log_messages = []
                for chunk in build_logs:
                    if 'stream' in chunk:
                        build_log_messages.append(chunk['stream'].strip())
                
                return {
                    'success': True,
                    'image_id': image.id,
                    'tag': tag,
                    'logs': build_log_messages
                }
                
            except docker.errors.BuildError as e:
                return {
                    'success': False,
                    'error': str(e),
                    'logs': [line.get('stream', '').strip() for line in e.build_log]
                }
    
    def run_container(self, image_tag: str, ports: Dict[str, str] = None, 
                     environment: Dict[str, str] = None) -> Dict[str, any]:
        """Executa container"""
        try:
            container = self.client.containers.run(
                image=image_tag,
                ports=ports,
                environment=environment,
                detach=True,
                remove=False  # Não remover automaticamente
            )
            
            return {
                'success': True,
                'container_id': container.id,
                'status': container.status
            }
            
        except docker.errors.APIError as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def stop_container(self, container_id: str) -> bool:
        """Para container"""
        try:
            container = self.client.containers.get(container_id)
            container.stop()
            container.remove()
            return True
        except docker.errors.NotFound:
            return False
        except docker.errors.APIError:
            return False
```

### Exemplo de Uso Integrado

```python
# src/examples/object_detection_example.py
from models.yolo_detector import YOLODetector
from models.optimization import ModelOptimizer
from deployment.containerizer import ModelContainerizer
import numpy as np
import cv2
import matplotlib.pyplot as plt

def demonstrate_object_detection():
    """Demonstra detecção de objetos com YOLO"""
    print("=== Demonstração de Detecção de Objetos ===\n")
    
    # Criar detector
    detector = YOLODetector()
    
    # Obter informações do modelo
    model_info = detector.get_model_info()
    print(f"Informações do modelo: {model_info}")
    
    # Criar imagem de exemplo
    sample_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    
    # Detectar objetos
    detections = detector.detect_objects(sample_image, confidence_threshold=0.5)
    
    print(f"\nDetecções encontradas: {len(detections)}")
    for i, detection in enumerate(detections):
        print(f"  Detecção {i+1}: {detection['class_name']} "
              f"(confiança: {detection['confidence']:.2f})")
    
    # Desenhar detecções
    result_image = detector.draw_detections(sample_image, detections)
    
    # Visualizar resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title(f'Detecções ({len(detections)} encontradas)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return detector, detections

def demonstrate_model_optimization():
    """Demonstra otimização de modelo"""
    print("\n=== Demonstração de Otimização de Modelo ===\n")
    
    # Criar modelo simples para demonstração
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Modelo original criado")
    print(f"Parâmetros: {model.count_params():,}")
    
    # Criar otimizador
    optimizer = ModelOptimizer(model)
    
    # Quantizar modelo
    print("\nQuantizando modelo...")
    quantized_model = optimizer.quantize_model()
    
    print(f"Tamanho do modelo original: {model.count_params() * 4 / (1024*1024):.2f} MB")
    print(f"Tamanho do modelo quantizado: {len(quantized_model) / (1024*1024):.2f} MB")
    
    # Criar dados de teste
    test_data = np.random.random((20, 224, 224, 3)).astype(np.float32)
    
    # Avaliar desempenho
    benchmark_results = optimizer.benchmark_model(quantized_model, test_data)
    
    print(f"\nResultados de benchmark:")
    print(f"  Tempo médio de inferência: {benchmark_results['avg_inference_time']:.4f}s")
    print(f"  FPS: {benchmark_results['fps']:.2f}")
    print(f"  Tamanho do modelo: {benchmark_results['model_size_mb']:.2f} MB")
    
    return optimizer, quantized_model

def demonstrate_containerization():
    """Demonstra containerização de modelo"""
    print("\n=== Demonstração de Containerização ===\n")
    
    # Criar containerizer
    containerizer = ModelContainerizer()
    
    # Criar conteúdo do Dockerfile
    dockerfile_content = containerizer.create_dockerfile_content(
        model_path="models/",
        requirements=[
            'tensorflow==2.15.0',
            'opencv-python==4.8.1.78',
            'numpy==1.24.3',
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0'
        ]
    )
    
    # Criar conteúdo do requirements.txt
    requirements_content = containerizer.create_requirements_txt([
        'tensorflow==2.15.0',
        'opencv-python==4.8.1.78',
        'numpy==1.24.3',
        'fastapi==0.104.1',
        'uvicorn[standard]==0.24.0'
    ])
    
    print("Dockerfile e requirements.txt criados")
    print("Para construir o container, você precisaria:")
    print("1. Ter os arquivos de modelo e código")
    print("2. Executar o build com os caminhos corretos")
    print("3. Configurar as portas e variáveis de ambiente")
    
    # Exibir conteúdo do Dockerfile
    print("\nConteúdo do Dockerfile:")
    print(dockerfile_content)
    
    return containerizer

def analyze_performance():
    """Analisa desempenho em diferentes condições"""
    print("\n=== Análise de Desempenho ===\n")
    
    # Simular diferentes tamanhos de modelo
    model_sizes_mb = [1, 5, 10, 25, 50, 100]
    original_times = []
    quantized_times = []
    
    for size in model_sizes_mb:
        # Simular tempo baseado no tamanho
        orig_time = size * 0.02  # Assumindo 0.02s por MB
        quant_time = orig_time * 0.6  # Quantização melhora em 40%
        
        original_times.append(orig_time)
        quantized_times.append(quant_time)
    
    # Plotar gráfico de comparação
    plt.figure(figsize=(10, 6))
    plt.plot(model_sizes_mb, original_times, 'o-', label='Modelo Original', linewidth=2)
    plt.plot(model_sizes_mb, quantized_times, 's-', label='Modelo Quantizado', linewidth=2)
    plt.xlabel('Tamanho do Modelo (MB)')
    plt.ylabel('Tempo de Inferência (s)')
    plt.title('Comparação de Desempenho: Modelo Original vs Quantizado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Análise mostrando benefícios da quantização para diferentes tamanhos de modelo")

if __name__ == "__main__":
    # Executar demonstrações
    detector, detections = demonstrate_object_detection()
    optimizer, quantized_model = demonstrate_model_optimization()
    containerizer = demonstrate_containerization()
    analyze_performance()
    
    print("\n=== Resumo da Aula ===")
    print("Hoje aprendemos:")
    print("- Detecção de objetos com YOLO")
    print("- Bounding boxes e confiança de detecções")
    print("- Quantização de modelos para otimização")
    print("- Containerização para deploy")
    print("- Análise de desempenho em diferentes condições")