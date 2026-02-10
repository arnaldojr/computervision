# Aula 3 - Segmentação e Detecção Baseada em Regras

## Objetivo da Aula

Implementar técnicas de segmentação e detecção de objetos baseadas em regras, criando funções reutilizáveis e desenvolvendo um sistema que comece a parecer um produto real.

## Conteúdo Teórico

### Threshold (Limiarização)

A limiarização é uma técnica fundamental de segmentação que converte uma imagem em escala de cinza em uma imagem binária, separando objetos de fundo com base em um valor de limiar.

Tipos comuns:
- **Limiar Global**: Um único valor para toda a imagem
- **Limiar Adaptativo**: Valores diferentes para diferentes regiões
- **Otsu's Method**: Método que automaticamente determina o melhor limiar

### Contornos

Contornos são curvas que conectam pontos contínuos de mesma intensidade, úteis para:
- Detecção de formas
- Análise de componentes conectados
- Extração de características de objetos

### Bounding Boxes

Retângulos que delimitam objetos detectados, fundamentais para:
- Localização de objetos
- Contagem de instâncias
- Extração de regiões de interesse

### Máscaras

Imagens binárias que indicam quais pixels pertencem a um objeto de interesse, usadas para:
- Segmentação de objetos
- Extração de regiões específicas
- Aplicação de operações apenas em áreas específicas

## Atividade Prática

### Implementar Técnicas de Threshold

```python
# src/features/thresholding.py
import cv2
import numpy as np

def global_threshold(image, threshold_value=127, max_value=255, method='binary'):
    """Aplica threshold global à imagem"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    if method == 'binary':
        _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    elif method == 'binary_inv':
        _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    elif method == 'truncate':
        _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TRUNC)
    elif method == 'tozero':
        _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO)
    elif method == 'tozero_inv':
        _, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO_INV)
    else:
        raise ValueError("Método de threshold não suportado")
    
    return binary

def adaptive_threshold(image, max_value=255, adaptive_method='mean', threshold_type='binary', block_size=11, c=2):
    """Aplica threshold adaptativo à imagem"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    if adaptive_method == 'mean':
        adaptive_method_cv = cv2.ADAPTIVE_THRESH_MEAN_C
    elif adaptive_method == 'gaussian':
        adaptive_method_cv = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError("Método adaptativo não suportado")
    
    if threshold_type == 'binary':
        threshold_type_cv = cv2.THRESH_BINARY
    elif threshold_type == 'binary_inv':
        threshold_type_cv = cv2.THRESH_BINARY_INV
    else:
        raise ValueError("Tipo de threshold não suportado")
    
    return cv2.adaptiveThreshold(gray, max_value, adaptive_method_cv, threshold_type_cv, block_size, c)

def otsu_threshold(image, max_value=255):
    """Aplica threshold de Otsu (automaticamente determina o melhor limiar)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    _, binary = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary, _
```

### Implementar Detecção de Contornos

```python
# src/features/contours.py
import cv2
import numpy as np

def find_contours(binary_image, retrieval_mode='external', approximation_method='simple'):
    """Encontra contornos em uma imagem binária"""
    if retrieval_mode == 'external':
        retrieval_mode_cv = cv2.RETR_EXTERNAL
    elif retrieval_mode == 'list':
        retrieval_mode_cv = cv2.RETR_LIST
    elif retrieval_mode == 'ccomp':
        retrieval_mode_cv = cv2.RETR_CCOMP
    elif retrieval_mode == 'tree':
        retrieval_mode_cv = cv2.RETR_TREE
    else:
        raise ValueError("Modo de recuperação não suportado")
    
    if approximation_method == 'none':
        approx_method_cv = cv2.CHAIN_APPROX_NONE
    elif approximation_method == 'simple':
        approx_method_cv = cv2.CHAIN_APPROX_SIMPLE
    elif approximation_method == 'tc89_l1':
        approx_method_cv = cv2.CHAIN_APPROX_TC89_L1
    elif approximation_method == 'tc89_kcos':
        approx_method_cv = cv2.CHAIN_APPROX_TC89_KCOS
    else:
        raise ValueError("Método de aproximação não suportado")
    
    contours, hierarchy = cv2.findContours(binary_image, retrieval_mode_cv, approx_method_cv)
    
    return contours, hierarchy

def filter_contours_by_area(contours, min_area=0, max_area=float('inf')):
    """Filtra contornos por área"""
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def filter_contours_by_circularity(contours, min_circularity=0, max_circularity=1):
    """Filtra contornos por circularidade (4*pi*area/perimeter^2)"""
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if min_circularity <= circularity <= max_circularity:
            filtered_contours.append(contour)
    
    return filtered_contours

def filter_contours_by_aspect_ratio(contours, min_ratio=0, max_ratio=float('inf')):
    """Filtra contornos por razão de aspecto (largura/altura do bounding rectangle)"""
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        if min_ratio <= aspect_ratio <= max_ratio:
            filtered_contours.append(contour)
    
    return filtered_contours

def get_contour_properties(contour):
    """Obtém propriedades de um contorno"""
    properties = {}
    
    # Área
    properties['area'] = cv2.contourArea(contour)
    
    # Perímetro
    properties['perimeter'] = cv2.arcLength(contour, True)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    properties['bounding_rect'] = {'x': x, 'y': y, 'width': w, 'height': h}
    
    # Bounding rectangle rotacionado
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    properties['rotated_rect'] = box
    
    # Circularity
    if properties['perimeter'] > 0:
        properties['circularity'] = 4 * np.pi * properties['area'] / (properties['perimeter'] * properties['perimeter'])
    else:
        properties['circularity'] = 0
    
    # Extent (razão entre área do contorno e área do bounding rectangle)
    properties['extent'] = properties['area'] / float(w * h)
    
    # Centroide
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        properties['centroid'] = (cx, cy)
    else:
        properties['centroid'] = (0, 0)
    
    return properties
```

### Implementar Bounding Boxes e Máscaras

```python
# src/features/bounding_boxes.py
import cv2
import numpy as np

def draw_bounding_boxes(image, contours, color=(0, 255, 0), thickness=2):
    """Desenha bounding boxes ao redor dos contornos"""
    result = image.copy()
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    
    return result

def draw_rotated_bounding_boxes(image, contours, color=(0, 255, 255), thickness=2):
    """Desenha bounding boxes rotacionadas ao redor dos contornos"""
    result = image.copy()
    
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, color, thickness)
    
    return result

def get_bounding_boxes(contours):
    """Obtém as coordenadas das bounding boxes para cada contorno"""
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
    
    return boxes

def create_masks_from_contours(image_shape, contours):
    """Cria máscaras binárias para cada contorno"""
    masks = []
    
    for contour in contours:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        masks.append(mask)
    
    return masks

def extract_roi(image, bounding_box):
    """Extrai região de interesse (ROI) baseada em bounding box"""
    x, y, w, h = bounding_box['x'], bounding_box['y'], bounding_box['width'], bounding_box['height']
    return image[y:y+h, x:x+w]

def create_combined_mask(image_shape, contours):
    """Cria uma máscara combinada para todos os contornos"""
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for contour in contours:
        cv2.fillPoly(combined_mask, [contour], 255)
    
    return combined_mask
```

### Detector de Objetos por Cor

```python
# src/features/color_detector.py
import cv2
import numpy as np

class ColorDetector:
    def __init__(self):
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],  # Precisa de dois ranges para vermelho
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
            'purple': [(130, 50, 50), (160, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)],
        }
    
    def detect_by_color_range(self, image, color_name, min_area=100):
        """Detecta objetos de uma cor específica"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        if color_name not in self.color_ranges:
            raise ValueError(f"Cor '{color_name}' não suportada")
        
        color_range = self.color_ranges[color_name]
        
        if color_name == 'red':
            # Vermelho tem dois ranges no HSV
            lower1, upper1 = color_range[0], color_range[1]
            lower2, upper2 = color_range[2], color_range[3]
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 + mask2
        else:
            lower, upper = color_range[0], color_range[1]
            mask = cv2.inRange(hsv, lower, upper)
        
        # Aplicar operações morfológicas para limpar a máscara
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar por área mínima
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        return filtered_contours, mask
    
    def detect_multiple_colors(self, image, colors, min_area=100):
        """Detecta múltiplas cores na mesma imagem"""
        results = {}
        
        for color in colors:
            contours, mask = self.detect_by_color_range(image, color, min_area)
            results[color] = {
                'contours': contours,
                'mask': mask,
                'count': len(contours)
            }
        
        return results
    
    def draw_color_detections(self, image, detection_results):
        """Desenha detecções de cores na imagem"""
        result_image = image.copy()
        
        # Cores para desenho (BGR)
        colors_bgr = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (128, 0, 128),
            'orange': (0, 165, 255)
        }
        
        for color_name, data in detection_results.items():
            contours = data['contours']
            color_bgr = colors_bgr.get(color_name, (255, 255, 255))  # Branco padrão
            
            # Desenhar contornos
            cv2.drawContours(result_image, contours, -1, color_bgr, 2)
            
            # Desenhar bounding boxes
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x+w, y+h), color_bgr, 2)
                
                # Adicionar texto com nome da cor e contagem
                cv2.putText(result_image, f'{color_name}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
        
        return result_image
```

### Sistema de Contagem Automática

```python
# src/features/object_counter.py
import cv2
import numpy as np

class ObjectCounter:
    def __init__(self):
        self.detection_history = []
    
    def count_objects(self, image, detection_method='contours', **kwargs):
        """Conta objetos na imagem usando diferentes métodos"""
        if detection_method == 'contours':
            # Converter para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Aplicar threshold
            threshold_value = kwargs.get('threshold_value', 127)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar por área mínima
            min_area = kwargs.get('min_area', 100)
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            
            return len(filtered_contours), filtered_contours
        
        elif detection_method == 'color':
            from .color_detector import ColorDetector
            detector = ColorDetector()
            
            color = kwargs.get('color', 'red')
            min_area = kwargs.get('min_area', 100)
            
            contours, _ = detector.detect_by_color_range(image, color, min_area)
            
            return len(contours), contours
        
        else:
            raise ValueError(f"Método de detecção '{detection_method}' não suportado")
    
    def count_and_save_results(self, image, output_path, detection_method='contours', **kwargs):
        """Conta objetos e salva resultados"""
        count, contours = self.count_objects(image, detection_method, **kwargs)
        
        # Desenhar resultados na imagem
        result_image = image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        
        # Adicionar texto com contagem
        cv2.putText(result_image, f'Contagem: {count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Salvar imagem
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Salvar informações
        result_info = {
            'count': count,
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'timestamp_not_available',
            'contour_areas': [cv2.contourArea(c) for c in contours],
            'image_path': output_path
        }
        
        self.detection_history.append(result_info)
        
        return result_info
```

### Exemplo de Uso Integrado

```python
# src/examples/segmentation_example.py
from features.thresholding import *
from features.contours import *
from features.bounding_boxes import *
from features.color_detector import ColorDetector
from features.object_counter import ObjectCounter
from utils.io import load_image_rgb, show_image
import matplotlib.pyplot as plt

def demonstrate_segmentation_techniques():
    """Demonstra técnicas de segmentação e detecção"""
    # Carregar imagem
    image = load_image_rgb("data/raw/exemplo.jpg")  # Substitua pelo caminho real
    
    # 1. Demonstrar diferentes tipos de threshold
    global_thresh = global_threshold(image, threshold_value=127)
    adaptive_thresh = adaptive_threshold(image)
    otsu_thresh, otsu_value = otsu_threshold(image)
    
    # 2. Encontrar contornos
    contours, hierarchy = find_contours(otsu_thresh)
    
    # 3. Filtrar contornos por área
    filtered_contours = filter_contours_by_area(contours, min_area=100)
    
    # 4. Desenhar bounding boxes
    bbox_image = draw_bounding_boxes(image, filtered_contours)
    
    # 5. Detector de cores
    color_detector = ColorDetector()
    color_results = color_detector.detect_multiple_colors(
        image, ['red', 'blue', 'green'], min_area=50
    )
    color_detected_image = color_detector.draw_color_detections(image, color_results)
    
    # 6. Contador de objetos
    counter = ObjectCounter()
    object_count, detected_contours = counter.count_objects(
        image, detection_method='contours', min_area=100
    )
    
    # Visualizar resultados
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].imshow(image)
    axes[0,0].set_title('Imagem Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(global_thresh, cmap='gray')
    axes[0,1].set_title('Threshold Global')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(adaptive_thresh, cmap='gray')
    axes[0,2].set_title('Threshold Adaptativo')
    axes[0,2].axis('off')
    
    axes[1,0].imshow(otsu_thresh, cmap='gray')
    axes[1,0].set_title(f'Threshold Otsu (v={otsu_value:.2f})')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(bbox_image)
    axes[1,1].set_title(f'Bounding Boxes (Contornos: {len(filtered_contours)})')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(color_detected_image)
    axes[1,2].set_title(f'Detecção por Cor (Objetos: {sum(r["count"] for r in color_results.values())})')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Contagem de objetos: {object_count}")
    print(f"Contornos detectados: {len(detected_contours)}")
    
    # Mostrar propriedades de alguns contornos
    if detected_contours:
        for i, contour in enumerate(detected_contours[:3]):  # Mostrar as primeiras 3
            props = get_contour_properties(contour)
            print(f"\nContorno {i+1}:")
            print(f"  Área: {props['area']:.2f}")
            print(f"  Circularidade: {props['circularity']:.3f}")
            print(f"  Razão de aspecto: {props['extent']:.3f}")
            print(f"  Centroide: {props['centroid']}")

if __name__ == "__main__":
    demonstrate_segmentation_techniques()
```

## Resultado Esperado

Nesta aula, você:

1. Implementou diferentes técnicas de threshold (global, adaptativo, Otsu)
2. Desenvolveu funções para detecção e filtragem de contornos
3. Criou funcionalidades para desenhar bounding boxes e extrair propriedades
4. Construiu um detector de objetos por cor
5. Desenvolveu um sistema de contagem automática de objetos
6. Integrar todas essas funcionalidades em um exemplo prático

O sistema agora começa a ter características de um produto real, com módulos bem definidos e funcionalidades que podem ser combinadas para resolver problemas específicos de detecção e segmentação.