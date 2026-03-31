# Aula 11 - Deploy de API e Modelos

## Objetivo da Aula

Implementar deploy de API e modelos de visão computacional, com foco em containerização, otimização de inferência e estratégias de disponibilização de modelos em produção.

## Conteúdo Teórico

### Estratégias de Deploy

Existem várias abordagens para deploy de modelos de visão computacional:

#### 1. Deploy em Nuvem
- **Vantagens**: Escalabilidade, gerenciamento facilitado, segurança
- **Desvantagens**: Latência, custos contínuos
- **Plataformas**: AWS SageMaker, Google Cloud ML Engine, Azure ML

#### 2. Deploy em Edge
- **Vantagens**: Baixa latência, privacidade, eficiência de banda
- **Desvantagens**: Recursos limitados, manutenção distribuída
- **Aplicações**: IoT, câmeras inteligentes, dispositivos móveis

#### 3. Deploy On-Premises
- **Vantagens**: Controle total, segurança de dados, custos previsíveis
- **Desvantagens**: Infraestrutura própria, manutenção
- **Aplicações**: Ambientes corporativos, setor público

### Containerização com Docker

Docker permite empacotar aplicações com todas as dependências necessárias, garantindo consistência entre ambientes de desenvolvimento, teste e produção.

### Otimização de Inferência

Técnicas para melhorar o desempenho de modelos em produção:

- **Quantização**: Redução da precisão numérica para diminuir tamanho e aumentar velocidade
- **Podas (Pruning)**: Remoção de conexões irrelevantes
- **Knowledge Distillation**: Criação de modelos menores que aprendem com modelos maiores
- **TensorRT (NVIDIA)**: Otimização para GPUs

### Monitoramento e Observabilidade

Fundamental para manter modelos em produção:
- **Métricas de desempenho**: Latência, throughput, acurácia
- **Logs**: Rastreamento de requisições e erros
- **Alertas**: Notificações de anomalias

## Atividade Prática

### Implementar Script de Deploy

```python
# src/deployment/deploy_manager.py
import docker
import os
import subprocess
import yaml
from typing import Dict, List, Optional
import json
import requests

class DeploymentManager:
    def __init__(self, config_file: str = None):
        self.client = docker.from_env()
        self.config = self._load_config(config_file) if config_file else {}
        self.containers = {}
    
    def _load_config(self, config_file: str) -> Dict:
        """Carrega configuração de deploy"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def build_image(self, dockerfile_path: str, context_path: str, image_name: str) -> bool:
        """Constroi imagem Docker"""
        try:
            print(f"Construindo imagem: {image_name}")
            image, build_logs = self.client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=image_name,
                rm=True
            )
            
            # Exibir logs de build
            for chunk in build_logs:
                if 'stream' in chunk:
                    print(chunk['stream'].strip())
            
            print(f"Imagem {image_name} construída com sucesso!")
            return True
            
        except docker.errors.BuildError as e:
            print(f"Erro ao construir imagem: {e}")
            for line in e.build_log:
                print(line.get('stream', ''))
            return False
    
    def run_container(self, image_name: str, container_name: str, 
                     ports: Dict[str, str] = None, 
                     environment: Dict[str, str] = None,
                     volumes: Dict[str, dict] = None) -> Optional[str]:
        """Executa container"""
        try:
            print(f"Executando container: {container_name}")
            
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                ports=ports,
                environment=environment,
                volumes=volumes,
                detach=True,
                auto_remove=False
            )
            
            self.containers[container_name] = container
            print(f"Container {container_name} iniciado com ID: {container.id}")
            return container.id
            
        except docker.errors.APIError as e:
            print(f"Erro ao executar container: {e}")
            return None
    
    def stop_container(self, container_name: str) -> bool:
        """Para e remove container"""
        try:
            if container_name in self.containers:
                container = self.containers[container_name]
                container.stop()
                container.remove()
                del self.containers[container_name]
                print(f"Container {container_name} parado e removido")
                return True
            else:
                print(f"Container {container_name} não encontrado")
                return False
        except docker.errors.APIError as e:
            print(f"Erro ao parar container: {e}")
            return False
    
    def check_health(self, container_name: str, health_endpoint: str = "/health") -> Dict:
        """Verifica saúde do serviço no container"""
        try:
            container = self.client.containers.get(container_name)
            port_bindings = container.attrs['NetworkSettings']['Ports']
            
            # Encontrar porta mapeada
            mapped_port = None
            for container_port, host_mapping in port_bindings.items():
                if container_port == '8000/tcp' and host_mapping:
                    mapped_port = host_mapping[0]['HostPort']
                    break
            
            if mapped_port:
                url = f"http://localhost:{mapped_port}{health_endpoint}"
                response = requests.get(url, timeout=5)
                
                return {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response_code': response.status_code,
                    'response': response.json() if response.content else {}
                }
            else:
                return {'status': 'unknown', 'error': 'Port not mapped'}
                
        except requests.exceptions.RequestException as e:
            return {'status': 'unhealthy', 'error': str(e)}
        except docker.errors.NotFound:
            return {'status': 'stopped', 'error': 'Container not found'}
    
    def scale_service(self, service_name: str, replicas: int):
        """Escala serviço (requer swarm mode ou kubernetes)"""
        # Esta é uma implementação simplificada
        # Em produção, usaria Kubernetes ou Docker Swarm
        print(f"Escalando serviço {service_name} para {replicas} réplicas")
        
        # Em uma implementação real, usaria:
        # - Kubernetes deployments
        # - Docker Swarm services
        # - AWS ECS
        pass
    
    def deploy_model_api(self, model_path: str, api_port: int = 8000) -> bool:
        """Implementa deploy completo de API de modelo"""
        image_name = "cv-model-api:latest"
        container_name = "cv-model-api-container"
        
        # 1. Criar Dockerfile
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
COPY {model_path} /app/models/

# Copiar código da aplicação
COPY app/ /app/app/
COPY src/ /app/src/

# Expor porta
EXPOSE {api_port}

# Comando para rodar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "{api_port}"]
"""
        
        # Salvar Dockerfile temporariamente
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Salvar requirements.txt
        requirements = [
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0',
            'tensorflow==2.15.0',
            'opencv-python==4.8.1.78',
            'numpy==1.24.3',
            'Pillow==10.1.0',
            'pydantic==2.5.0',
            'pydantic-settings==2.1.0',
            'python-multipart==0.0.6'
        ]
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))
        
        # 2. Construir imagem
        if not self.build_image('Dockerfile', '.', image_name):
            return False
        
        # 3. Executar container
        ports = {f'{api_port}/tcp': api_port}
        environment = {
            'MODEL_PATH': f'/app/models/{os.path.basename(model_path)}',
            'DEBUG': 'false'
        }
        
        container_id = self.run_container(
            image_name=image_name,
            container_name=container_name,
            ports=ports,
            environment=environment
        )
        
        if not container_id:
            return False
        
        # 4. Verificar saúde
        import time
        time.sleep(10)  # Aguardar inicialização
        
        health_status = self.check_health(container_name)
        print(f"Status de saúde: {health_status}")
        
        return health_status['status'] == 'healthy'
```

### Implementar Otimização de Inferência

```python
# src/deployment/inference_optimizer.py
import tensorflow as tf
import numpy as np
from typing import Any, Dict, List
import time

class InferenceOptimizer:
    def __init__(self, model):
        self.model = model
        self.optimized_model = None
    
    def quantize_model(self) -> bytes:
        """Quantiza modelo para inferência otimizada"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        return quantized_model
    
    def quantize_int8(self, representative_dataset) -> bytes:
        """Quantiza modelo para INT8 com dataset representativo"""
        def representative_data_gen():
            for batch in representative_dataset.take(100):  # Usar 100 batches
                yield [batch]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        quantized_model = converter.convert()
        return quantized_model
    
    def create_tensorrt_model(self, input_shape: tuple) -> Any:
        """Cria modelo otimizado com TensorRT (requer GPU NVIDIA)"""
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.FP16,
                max_workspace_size_bytes=8000000000  # 8GB
            )
            
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=None,  # Caminho para modelo salvo
                conversion_params=conversion_params
            )
            
            # Esta é uma implementação conceitual
            # Em prática, precisaria de um modelo salvo
            print("TensorRT optimization requires a saved model")
            return None
            
        except ImportError:
            print("TensorRT não disponível")
            return None
    
    def benchmark_models(self, original_model, tflite_model_bytes, 
                        test_data: np.ndarray, num_runs: int = 100) -> Dict[str, Any]:
        """Compara desempenho de modelos"""
        # Benchmark modelo original
        original_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = original_model.predict(test_data[:1])
            original_times.append(time.time() - start)
        
        # Benchmark modelo TFLite
        interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        tflite_times = []
        for _ in range(num_runs):
            # Preparar entrada
            input_data = test_data[:1].astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            start = time.time()
            interpreter.invoke()
            tflite_times.append(time.time() - start)
        
        return {
            'original': {
                'avg_time': np.mean(original_times),
                'std_time': np.std(original_times),
                'min_time': np.min(original_times),
                'max_time': np.max(original_times),
                'throughput': len(test_data) / sum(original_times)
            },
            'tflite': {
                'avg_time': np.mean(tflite_times),
                'std_time': np.std(tflite_times),
                'min_time': np.min(tflite_times),
                'max_time': np.max(tflite_times),
                'throughput': len(test_data) / sum(tflite_times)
            },
            'improvement': np.mean(original_times) / np.mean(tflite_times)
        }
    
    def optimize_for_device(self, device_type: str = 'mobile') -> bytes:
        """Otimiza modelo para dispositivo específico"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if device_type == 'mobile':
            # Otimizações para dispositivos móveis
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]  # Meia precisão
        elif device_type == 'edge':
            # Otimizações para dispositivos edge
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS  # Para operações não suportadas
            ]
        elif device_type == 'server':
            # Otimizações para servidores
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Pode adicionar outras otimizações específicas
        
        optimized_model = converter.convert()
        return optimized_model
    
    def create_optimized_pipeline(self, model, input_shape: tuple, 
                                 device_target: str = 'mobile') -> Any:
        """Cria pipeline otimizado para inferência"""
        # Converter para TensorFlow Lite otimizado
        tflite_model = self.optimize_for_device(device_target)
        
        # Carregar modelo otimizado
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Obter detalhes de entrada e saída
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        class OptimizedModel:
            def __init__(self, interpreter, input_details, output_details):
                self.interpreter = interpreter
                self.input_details = input_details
                self.output_details = output_details
            
            def predict(self, input_data):
                # Preparar entrada
                input_data = input_data.astype(np.float32)
                
                # Definir tensor de entrada
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                
                # Executar inferência
                self.interpreter.invoke()
                
                # Obter saída
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                return output
        
        return OptimizedModel(interpreter, input_details, output_details)
```

### Implementar Monitoramento

```python
# src/deployment/monitoring.py
import psutil
import GPUtil
import time
import threading
from typing import Dict, Callable
import json
import requests
from datetime import datetime

class ModelMonitor:
    def __init__(self, api_endpoint: str = "http://localhost:8000"):
        self.api_endpoint = api_endpoint
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'request_latency': [],
            'throughput': [],
            'error_rate': []
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Inicia monitoramento em thread separada"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
        print("Monitoramento iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Monitoramento parado")
    
    def _monitor_loop(self, interval: float):
        """Loop de monitoramento"""
        while self.monitoring:
            # Coletar métricas do sistema
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Coletar métricas de GPU se disponível
            gpu_percent = 0
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
            
            # Adicionar métricas
            self.metrics['cpu_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': cpu_percent
            })
            
            self.metrics['memory_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': memory_percent
            })
            
            self.metrics['gpu_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': gpu_percent
            })
            
            time.sleep(interval)
    
    def test_api_performance(self, endpoint: str, num_requests: int = 100) -> Dict:
        """Testa desempenho da API"""
        latencies = []
        errors = 0
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                # Fazer requisição de teste (exemplo para endpoint de classificação)
                response = requests.post(
                    f"{self.api_endpoint}{endpoint}",
                    json={'image_base64': 'dummy'}  # Isso falhará, mas mede latência
                )
                latency = time.time() - start_time
                latencies.append(latency)
                
                if response.status_code != 200:
                    errors += 1
                    
            except requests.exceptions.RequestException:
                errors += 1
                latencies.append(time.time() - start_time)  # Contar erro na latência também
        
        return {
            'avg_latency': np.mean(latencies) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
            'error_rate': errors / num_requests if num_requests > 0 else 0,
            'throughput': num_requests / sum(latencies) if latencies and sum(latencies) > 0 else 0
        }
    
    def get_current_metrics(self) -> Dict:
        """Obtém métricas atuais"""
        current_cpu = psutil.cpu_percent(interval=1)
        current_memory = psutil.virtual_memory().percent
        
        gpus = GPUtil.getGPUs()
        current_gpu = gpus[0].load * 100 if gpus else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': current_cpu,
            'memory_usage': current_memory,
            'gpu_usage': current_gpu,
            'active_monitoring': self.monitoring
        }
    
    def export_metrics(self, filename: str = 'metrics.json'):
        """Exporta métricas para arquivo"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"Métricas exportadas para {filename}")
    
    def setup_alerts(self, thresholds: Dict[str, float], callback: Callable):
        """Configura sistema de alertas"""
        # Este é um exemplo simplificado
        # Em produção, usaria um sistema mais robusto
        def alert_checker():
            while self.monitoring:
                current_metrics = self.get_current_metrics()
                
                for metric, threshold in thresholds.items():
                    if metric in current_metrics and current_metrics[metric] > threshold:
                        alert_msg = f"ALERTA: {metric} excedeu threshold! Valor: {current_metrics[metric]}, Limite: {threshold}"
                        callback(alert_msg)
                
                time.sleep(10)  # Verificar a cada 10 segundos
        
        alert_thread = threading.Thread(target=alert_checker)
        alert_thread.daemon = True
        alert_thread.start()
```

### Exemplo de Uso Completo

```python
# src/examples/deployment_example.py
from deployment.deploy_manager import DeploymentManager
from deployment.inference_optimizer import InferenceOptimizer
from deployment.monitoring import ModelMonitor
import tensorflow as tf
import numpy as np

def demonstrate_full_deployment():
    """Demonstra deploy completo de modelo de CV"""
    print("=== Demonstração de Deploy Completo ===\n")
    
    # 1. Criar modelo de exemplo
    print("Criando modelo de exemplo...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Modelo criado com sucesso!")
    
    # 2. Otimizar modelo para inferência
    print("\nOtimizando modelo para inferência...")
    optimizer = InferenceOptimizer(model)
    
    # Quantizar modelo
    quantized_model = optimizer.quantize_model()
    print(f"Modelo quantizado: {len(quantized_model) / (1024*1024):.2f} MB")
    
    # Criar dados de teste
    test_data = np.random.random((10, 224, 224, 3)).astype(np.float32)
    
    # Benchmark
    benchmark_results = optimizer.benchmark_models(model, quantized_model, test_data, num_runs=10)
    
    print(f"\nResultados de benchmark:")
    print(f"  Modelo original - Média: {benchmark_results['original']['avg_time']:.4f}s")
    print(f"  Modelo TFLite - Média: {benchmark_results['tflite']['avg_time']:.4f}s")
    print(f"  Melhoria: {benchmark_results['improvement']:.2f}x")
    
    # 3. Preparar para deploy
    print("\nPreparando para deploy...")
    
    # Criar deploy manager
    deploy_manager = DeploymentManager()
    
    # Nota: O deploy real requer arquivos reais e infraestrutura
    # Esta é uma demonstração conceitual
    print("Deploy manager inicializado")
    
    # 4. Configurar monitoramento
    print("\nConfigurando monitoramento...")
    monitor = ModelMonitor(api_endpoint="http://localhost:8000")
    
    # Iniciar monitoramento
    monitor.start_monitoring(interval=2.0)
    
    # Obter métricas atuais
    current_metrics = monitor.get_current_metrics()
    print(f"Métricas atuais: {current_metrics}")
    
    # Simular teste de performance
    perf_results = monitor.test_api_performance('/vision/classify', num_requests=10)
    print(f"Resultados de performance simulada: {perf_results}")
    
    # Parar monitoramento
    monitor.stop_monitoring()
    
    print("\nDemonstração de deploy concluída!")
    print("Próximos passos para deploy real:")
    print("1. Preparar Dockerfile com modelo e dependências")
    print("2. Construir imagem Docker")
    print("3. Executar container com configurações de produção")
    print("4. Configurar balanceamento de carga e escalabilidade")
    print("5. Implementar monitoramento contínuo")
    print("6. Configurar alertas e recuperação de falhas")

def compare_deployment_strategies():
    """Compara diferentes estratégias de deployment"""
    strategies = {
        'Nuvem': {
            'vantagens': ['Escalabilidade automática', 'Gerenciamento facilitado', 'Alta disponibilidade'],
            'desvantagens': ['Latência de rede', 'Custos contínuos', 'Dependência de provedor'],
            'uso': 'Aplicações com demanda variável'
        },
        'Edge': {
            'vantagens': ['Baixa latência', 'Privacidade dos dados', 'Eficiência de banda'],
            'desvantagens': ['Recursos limitados', 'Difícil manutenção', 'Atualizações complexas'],
            'uso': 'IoT, câmeras inteligentes, dispositivos móveis'
        },
        'On-Premises': {
            'vantagens': ['Controle total', 'Segurança de dados', 'Custos previsíveis'],
            'desvantagens': ['Infraestrutura própria', 'Manutenção', 'Escalabilidade limitada'],
            'uso': 'Ambientes corporativos, setor público'
        }
    }
    
    print("\n=== Comparação de Estratégias de Deployment ===\n")
    
    for strategy, details in strategies.items():
        print(f"{strategy}:")
        print(f"  Vantagens: {', '.join(details['vantagens'])}")
        print(f"  Desvantagens: {', '.join(details['desvantagens'])}")
        print(f"  Uso recomendado: {details['uso']}\n")

def optimize_for_production():
    """Demonstra otimizações para produção"""
    print("=== Otimizações para Produção ===\n")
    
    optimizations = [
        {
            'nome': 'Quantização',
            'descricao': 'Redução da precisão numérica para diminuir tamanho e aumentar velocidade',
            'ganho': '2-4x redução de tamanho, 2-3x aumento de velocidade'
        },
        {
            'nome': 'Podas (Pruning)', 
            'descricao': 'Remoção de conexões irrelevantes para reduzir complexidade',
            'ganho': 'Até 50% redução de parâmetros com mínima perda de acurácia'
        },
        {
            'nome': 'TensorRT (NVIDIA)',
            'descricao': 'Otimização específica para GPUs NVIDIA',
            'ganho': '2-7x aumento de velocidade em GPUs compatíveis'
        },
        {
            'nome': 'Model Distillation',
            'descricao': 'Criação de modelos menores que aprendem com modelos maiores',
            'ganho': 'Modelos 10-100x menores com desempenho similar'
        }
    ]
    
    for opt in optimizations:
        print(f"{opt['nome']}:")
        print(f"  Descrição: {opt['descricao']}")
        print(f"  Ganho esperado: {opt['ganho']}\n")

if __name__ == "__main__":
    # Executar demonstrações
    demonstrate_full_deployment()
    compare_deployment_strategies()
    optimize_for_production()
    
    print("\n=== Resumo da Aula ===")
    print("Hoje aprendemos:")
    print("- Estratégias de deploy para modelos de CV")
    print("- Containerização com Docker")
    print("- Otimização de inferência")
    print("- Monitoramento e observabilidade")
    print("- Comparação de abordagens de deploy")
    print("- Técnicas de otimização para produção")