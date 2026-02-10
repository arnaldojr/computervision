# CP1 - Pipeline de Processamento

## Objetivo

Criar um pipeline funcional de processamento de imagem que receba uma imagem, aplique técnicas de pré-processamento e segmentação, detecte objetos e retorne bounding boxes.

## Requisitos Técnicos

Seu projeto deve incluir:

1. **Estrutura de projeto organizada** seguindo a arquitetura vista nas aulas anteriores
2. **Pipeline de processamento** que realize múltiplas etapas de transformação
3. **Sistema de detecção** baseado em regras (threshold, contornos, etc.)
4. **Visualização dos resultados** com bounding boxes
5. **Documentação básica** do código e do processo

## Etapas do Projeto

### 1. Estruturação do Projeto

Sua estrutura de projeto deve seguir o padrão estabelecido:

```
cp1_pipeline/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── filters.py
│   │   ├── morphology.py
│   │   └── pipeline.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── thresholding.py
│   │   ├── contours.py
│   │   └── bounding_boxes.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── io.py
│   └── main.py
├── data/
│   ├── input/
│   └── output/
├── tests/
├── requirements.txt
├── README.md
└── config.yaml
```

### 2. Implementação do Pipeline

Crie um pipeline que combine as técnicas aprendidas:

```python
# src/main.py
from preprocessing.pipeline import ImagePreprocessingPipeline
from features.thresholding import otsu_threshold
from features.contours import find_contours, filter_contours_by_area
from features.bounding_boxes import draw_bounding_boxes
from utils.io import load_image_rgb, show_image
import cv2
import os

def create_cv_pipeline():
    """Cria um pipeline completo de visão computacional"""
    pipeline = ImagePreprocessingPipeline()
    
    # Adiciona etapas ao pipeline
    pipeline.add_grayscale() \
             .add_histogram_equalization(color_space='grayscale') \
             .add_blur(blur_type='gaussian', kernel_size=(3, 3)) \
             .add_edge_detection(edge_type='canny', low_threshold=50, high_threshold=150)
    
    return pipeline

def process_image_pipeline(input_path, output_path):
    """Processa uma imagem usando o pipeline completo"""
    # Carregar imagem
    image = load_image_rgb(input_path)
    
    # Criar e aplicar pipeline
    pipeline = create_cv_pipeline()
    processed_image = pipeline.process(image)
    
    # Aplicar threshold para binarização
    binary_image, _ = otsu_threshold(processed_image)
    
    # Encontrar contornos
    contours, _ = find_contours(binary_image)
    
    # Filtrar contornos por área
    filtered_contours = filter_contours_by_area(contours, min_area=100)
    
    # Desenhar bounding boxes
    result_image = draw_bounding_boxes(image, filtered_contours)
    
    # Salvar resultado
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    return result_image, len(filtered_contours)

def main():
    """Função principal para executar o pipeline"""
    input_path = "data/input/exemplo.jpg"  # Caminho da imagem de entrada
    output_path = "data/output/result.jpg"  # Caminho para salvar resultado
    
    # Verificar se a imagem de entrada existe
    if not os.path.exists(input_path):
        print(f"Erro: Imagem de entrada não encontrada em {input_path}")
        return
    
    # Processar imagem
    result_image, object_count = process_image_pipeline(input_path, output_path)
    
    print(f"Processamento concluído!")
    print(f"Objetos detectados: {object_count}")
    print(f"Resultado salvo em: {output_path}")
    
    # Mostrar resultado
    show_image(result_image, f"Resultado - {object_count} objetos detectados")

if __name__ == "__main__":
    main()
```

### 3. Configuração do Projeto

```yaml
# config.yaml
pipeline:
  grayscale: true
  histogram_equalization: true
  blur:
    enabled: true
    type: gaussian
    kernel_size: [3, 3]
  edge_detection:
    enabled: true
    type: canny
    low_threshold: 50
    high_threshold: 150

segmentation:
  threshold_method: otsu
  min_contour_area: 100
  max_contour_area: 100000

output:
  bounding_box_color: [0, 255, 0]
  bounding_box_thickness: 2
```

### 4. Script de Teste

```python
# tests/test_cp1.py
import unittest
import cv2
import numpy as np
import os
from src.main import process_image_pipeline

class TestCPPipeline(unittest.TestCase):
    def setUp(self):
        """Configuração antes de cada teste"""
        # Criar imagem de teste
        self.test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite('data/input/test_image.jpg', cv2.cvtColor(self.test_image, cv2.COLOR_RGB2BGR))
    
    def test_process_image_pipeline(self):
        """Testa o pipeline completo de processamento"""
        input_path = 'data/input/test_image.jpg'
        output_path = 'data/output/test_result.jpg'
        
        # Executar pipeline
        result_image, object_count = process_image_pipeline(input_path, output_path)
        
        # Verificar se o resultado foi salvo
        self.assertTrue(os.path.exists(output_path))
        
        # Verificar se a imagem de resultado tem dimensões válidas
        self.assertEqual(len(result_image.shape), 3)  # RGB
        
        # Verificar se algum objeto foi detectado (pode variar com imagem aleatória)
        self.assertIsInstance(object_count, int)
        self.assertGreaterEqual(object_count, 0)
    
    def tearDown(self):
        """Limpeza após cada teste"""
        test_files = [
            'data/input/test_image.jpg',
            'data/output/test_result.jpg'
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
```

### 5. Documentação

```markdown
# CP1 - Pipeline de Processamento

Este projeto implementa um pipeline completo de processamento de imagem para detecção de objetos.

## Funcionalidades

- Pré-processamento de imagem (equalização, desfoque, detecção de bordas)
- Segmentação por threshold
- Detecção de contornos
- Desenho de bounding boxes
- Contagem de objetos

## Como usar

1. Coloque sua imagem em `data/input/`
2. Execute `python src/main.py`
3. O resultado será salvo em `data/output/`

## Configuração

As configurações do pipeline podem ser ajustadas em `config.yaml`.
```

## Critérios de Avaliação

Seu projeto será avaliado com base nos seguintes critérios:

- **Correção técnica** (40%): O código funciona corretamente e produz resultados esperados
- **Organização do código** (25%): Estrutura bem organizada, modular e seguindo boas práticas
- **Funcionalidade do pipeline** (20%): Implementação completa do pipeline com múltiplas etapas
- **Documentação** (15%): Código e projeto devidamente documentados

## Entrega

A entrega pode ser feita como:

- Script organizado com todas as funcionalidades implementadas
- Pequena API local (opcional) que aceita upload de imagem e retorna resultado
- Demonstração funcional do pipeline

Lembre-se de que o foco está na implementação de um pipeline funcional que demonstre compreensão das técnicas de visão computacional vistas até agora.