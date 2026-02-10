# Aula 2 - Processamento de Imagem Aplicado

## Objetivo da Aula

Aplicar técnicas de processamento de imagem em um contexto prático, criando funções reutilizáveis e organização modular para o pipeline de visão computacional.

## Conteúdo Teórico

### Espaços de Cor

O espaço de cor define como as cores são representadas em uma imagem digital. Cada espaço tem suas vantagens para diferentes tipos de operações:

- **RGB (Red, Green, Blue)**: Modelo aditivo baseado na combinação de luz vermelha, verde e azul. Adequado para dispositivos de exibição.
- **HSV (Hue, Saturation, Value)**: Separa cor, saturação e brilho. Ideal para operações baseadas em cor específica.
- **LAB**: Separa luminância de cor, útil para correções de cor independentes da iluminação.

### Histogramas

O histograma de uma imagem mostra a distribuição de intensidades de pixel. É útil para análise e correção de contraste, permitindo técnicas como equalização de histograma.

### Filtros e Convolução

A convolução é uma operação fundamental que aplica um kernel (filtro) sobre uma imagem. Tipos comuns:

- **Filtros Passa-Baixa**: Suavizam a imagem, reduzindo ruído (média, gaussiano, mediana)
- **Filtros Passa-Alta**: Realçam bordas e detalhes (Sobel, Laplaciano, Canny)

### Operações Morfológicas

Operações que manipulam a estrutura geométrica dos objetos em uma imagem binária:

- **Erosão**: Reduz o tamanho dos objetos brancos
- **Dilatação**: Aumenta o tamanho dos objetos brancos
- **Abertura**: Erosão seguida de dilatação (remove ruído)
- **Fechamento**: Dilatação seguida de erosão (fecha lacunas)

## Atividade Prática

### Implementar Pipeline de Pré-processamento

Vamos expandir nosso projeto com módulos específicos para pré-processamento:

```python
# src/preprocessing/color_spaces.py
import cv2
import numpy as np

def rgb_to_hsv(image):
    """Converte imagem de RGB para HSV"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def rgb_to_grayscale(image):
    """Converte imagem de RGB para escala de cinza"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def adjust_brightness_contrast(image, brightness=0, contrast=1):
    """Ajusta brilho e contraste da imagem"""
    # Primeiro ajustar contraste, depois brilho
    adjusted = image * contrast + brightness
    # Garantir que os valores estejam entre 0 e 255
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

def equalize_histogram(image):
    """Equaliza histograma de imagem em escala de cinza"""
    if len(image.shape) == 3:
        # Se for imagem colorida, converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized
    else:
        return cv2.equalizeHist(image)

def equalize_histogram_color(image):
    """Equaliza histograma de imagem colorida (canal V do HSV)"""
    # Converter para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Equalizar canal V (valor/brilho)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    
    # Converter de volta para RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

### Implementar Filtros e Convolução

```python
# src/preprocessing/filters.py
import cv2
import numpy as np

def apply_blur(image, kernel_size=(5, 5)):
    """Aplica filtro de desfoque"""
    return cv2.blur(image, kernel_size)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    """Aplica filtro de desfoque gaussiano"""
    return cv2.GaussianBlur(image, kernel_size, sigma_x)

def apply_median_blur(image, kernel_size=5):
    """Aplica filtro de mediana (bom para remover ruído sal e pimenta)"""
    return cv2.medianBlur(image, kernel_size)

def detect_edges_sobel(image):
    """Detecta bordas usando o operador Sobel"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Gradientes X e Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude do gradiente
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Converter de volta para uint8
    sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    
    return sobel_combined

def detect_edges_laplacian(image):
    """Detecta bordas usando o operador Laplaciano"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Converter de volta para uint8
    laplacian = np.uint8(np.absolute(laplacian))
    
    return laplacian

def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """Detecta bordas usando o detector de Canny"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    return cv2.Canny(gray, low_threshold, high_threshold)

def apply_custom_filter(image, kernel):
    """Aplica filtro personalizado usando convolução"""
    if len(image.shape) == 3:
        # Aplicar filtro a cada canal
        filtered = np.zeros_like(image)
        for i in range(image.shape[2]):
            filtered[:,:,i] = cv2.filter2D(image[:,:,i], -1, kernel)
        return filtered
    else:
        return cv2.filter2D(image, -1, kernel)
```

### Implementar Operações Morfológicas

```python
# src/preprocessing/morphology.py
import cv2
import numpy as np

def create_structuring_element(shape='rect', size=5):
    """Cria elemento estruturante para operações morfológicas"""
    if shape == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        raise ValueError("Forma não suportada. Use 'rect', 'ellipse' ou 'cross'")

def morphological_erosion(image, kernel_size=5, iterations=1):
    """Aplica erosão morfológica"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)

def morphological_dilation(image, kernel_size=5, iterations=1):
    """Aplica dilatação morfológica"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)

def morphological_opening(image, kernel_size=5, iterations=1):
    """Aplica abertura morfológica (erosão seguida de dilatação)"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def morphological_closing(image, kernel_size=5, iterations=1):
    """Aplica fechamento morfológico (dilatação seguida de erosão)"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def morphological_gradient(image, kernel_size=5):
    """Aplica gradiente morfológico"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def top_hat(image, kernel_size=5):
    """Aplica transformada Top Hat"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def black_hat(image, kernel_size=5):
    """Aplica transformada Black Hat"""
    kernel = create_structuring_element(size=kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
```

### Pipeline de Pré-processamento Integrado

```python
# src/preprocessing/pipeline.py
from .color_spaces import *
from .filters import *
from .morphology import *
import numpy as np

class ImagePreprocessingPipeline:
    def __init__(self):
        self.steps = []
    
    def add_resize(self, width, height):
        """Adiciona passo de redimensionamento"""
        def resize_step(image):
            return cv2.resize(image, (width, height))
        self.steps.append(('resize', resize_step))
        return self
    
    def add_grayscale(self):
        """Adiciona passo de conversão para escala de cinza"""
        self.steps.append(('grayscale', rgb_to_grayscale))
        return self
    
    def add_color_conversion(self, conversion_func):
        """Adiciona passo de conversão de espaço de cor"""
        self.steps.append(('color_conversion', conversion_func))
        return self
    
    def add_brightness_contrast(self, brightness=0, contrast=1):
        """Adiciona passo de ajuste de brilho e contraste"""
        def bc_step(image):
            return adjust_brightness_contrast(image, brightness, contrast)
        self.steps.append(('brightness_contrast', bc_step))
        return self
    
    def add_blur(self, blur_type='gaussian', kernel_size=(5, 5)):
        """Adiciona passo de desfoque"""
        if blur_type == 'gaussian':
            blur_func = lambda img: apply_gaussian_blur(img, kernel_size)
        elif blur_type == 'average':
            blur_func = lambda img: apply_blur(img, kernel_size)
        elif blur_type == 'median':
            blur_func = lambda img: apply_median_blur(img, kernel_size[0])
        else:
            raise ValueError("Tipo de desfoque não suportado")
        
        self.steps.append(('blur', blur_func))
        return self
    
    def add_edge_detection(self, edge_type='canny', **kwargs):
        """Adiciona passo de detecção de bordas"""
        if edge_type == 'sobel':
            edge_func = detect_edges_sobel
        elif edge_type == 'laplacian':
            edge_func = detect_edges_laplacian
        elif edge_type == 'canny':
            def canny_wrapper(image):
                low = kwargs.get('low_threshold', 50)
                high = kwargs.get('high_threshold', 150)
                return detect_edges_canny(image, low, high)
            edge_func = canny_wrapper
        else:
            raise ValueError("Tipo de detecção de bordas não suportado")
        
        self.steps.append(('edge_detection', edge_func))
        return self
    
    def add_morphological_operation(self, operation, kernel_size=5, **kwargs):
        """Adiciona operação morfológica"""
        if operation == 'erosion':
            morph_func = lambda img: morphological_erosion(img, kernel_size, kwargs.get('iterations', 1))
        elif operation == 'dilation':
            morph_func = lambda img: morphological_dilation(img, kernel_size, kwargs.get('iterations', 1))
        elif operation == 'opening':
            morph_func = lambda img: morphological_opening(img, kernel_size, kwargs.get('iterations', 1))
        elif operation == 'closing':
            morph_func = lambda img: morphological_closing(img, kernel_size, kwargs.get('iterations', 1))
        elif operation == 'gradient':
            morph_func = lambda img: morphological_gradient(img, kernel_size)
        else:
            raise ValueError("Operação morfológica não suportada")
        
        self.steps.append(('morphological', morph_func))
        return self
    
    def add_histogram_equalization(self, color_space='grayscale'):
        """Adiciona equalização de histograma"""
        if color_space == 'grayscale':
            eq_func = equalize_histogram
        elif color_space == 'color':
            eq_func = equalize_histogram_color
        else:
            raise ValueError("Espaço de cor para equalização não suportado")
        
        self.steps.append(('histogram_equalization', eq_func))
        return self
    
    def process(self, image):
        """Processa imagem aplicando todos os passos do pipeline"""
        result = image.copy()
        
        for step_name, step_func in self.steps:
            try:
                result = step_func(result)
            except Exception as e:
                print(f"Erro no passo '{step_name}': {str(e)}")
                # Continuar com a próxima etapa ou retornar imagem anterior
                continue
        
        return result
    
    def reset(self):
        """Reseta o pipeline removendo todos os passos"""
        self.steps = []
        return self
```

### Exemplo de Uso do Pipeline

```python
# src/examples/preprocessing_example.py
from preprocessing.pipeline import ImagePreprocessingPipeline
from utils.io import load_image_rgb, show_image
import matplotlib.pyplot as plt

def demonstrate_preprocessing_pipeline():
    """Demonstra o uso do pipeline de pré-processamento"""
    # Carregar imagem
    image = load_image_rgb("data/raw/exemplo.jpg")  # Substitua pelo caminho real
    
    # Criar diferentes pipelines
    # Pipeline 1: Ajuste de brilho e contraste + desfoque gaussiano
    pipeline1 = ImagePreprocessingPipeline()
    pipeline1.add_brightness_contrast(brightness=20, contrast=1.2) \
             .add_blur(blur_type='gaussian', kernel_size=(5, 5))
    
    # Pipeline 2: Detecção de bordas Canny
    pipeline2 = ImagePreprocessingPipeline()
    pipeline2.add_grayscale() \
             .add_edge_detection(edge_type='canny', low_threshold=50, high_threshold=150)
    
    # Pipeline 3: Equalização de histograma + operações morfológicas
    pipeline3 = ImagePreprocessingPipeline()
    pipeline3.add_histogram_equalization(color_space='color') \
             .add_morphological_operation('opening', kernel_size=3, iterations=1) \
             .add_morphological_operation('closing', kernel_size=3, iterations=1)
    
    # Processar imagens
    processed1 = pipeline1.process(image)
    processed2 = pipeline2.process(image)
    processed3 = pipeline3.process(image)
    
    # Visualizar resultados
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0,0].imshow(image)
    axes[0,0].set_title('Imagem Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(processed1)
    axes[0,1].set_title('Brilho/Contraste + Blur')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(processed2, cmap='gray')
    axes[1,0].set_title('Detecção de Bordas (Canny)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(processed3)
    axes[1,1].set_title('Equalização + Morfológico')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Limpar pipelines
    pipeline1.reset()
    pipeline2.reset()
    pipeline3.reset()

if __name__ == "__main__":
    demonstrate_preprocessing_pipeline()
```

## Resultado Esperado

Nesta aula, você:

1. Aprendeu sobre diferentes espaços de cor e suas aplicações
2. Implementou funções para manipulação de histogramas
3. Criou filtros e operadores de detecção de bordas
4. Desenvolveu operações morfológicas
5. Construiu um pipeline de pré-processamento modular e reutilizável
6. Testou diferentes combinações de operações para ver seus efeitos

O pipeline modular permite fácil experimentação e combinação de diferentes técnicas de pré-processamento, facilitando a criação de soluções específicas para diferentes problemas de visão computacional.