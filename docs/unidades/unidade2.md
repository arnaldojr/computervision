# Unidade 2 - Processamento e Segmentação

## Espaços de Cor

O espaço de cor define como as cores são representadas em uma imagem digital. Cada espaço de cor tem suas vantagens para diferentes tipos de operações de processamento de imagem.

### RGB (Red, Green, Blue)
- Modelo aditivo baseado na combinação de luz vermelha, verde e azul
- Adequado para dispositivos de exibição
- Limitações para separação de cor e brilho

```python
import cv2
import matplotlib.pyplot as plt

# Carregar imagem
img = cv2.imread('imagem.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Visualizar canais RGB separadamente
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(img_rgb)
axes[0].set_title('Imagem Original')
axes[1].imshow(img_rgb[:,:,0], cmap='Reds_r')
axes[1].set_title('Canal R')
axes[2].imshow(img_rgb[:,:,1], cmap='Greens_r')
axes[2].set_title('Canal G')
axes[3].imshow(img_rgb[:,:,2], cmap='Blues_r')
axes[3].set_title('Canal B')
plt.show()
```

### HSV (Hue, Saturation, Value)
- Hue (Matiz): Cor pura (0-360°)
- Saturation (Saturação): Pureza da cor (0-100%)
- Value (Valor): Brilho (0-100%)

Ideal para operações baseadas em cor, como detecção de objetos por cor específica.

```python
import cv2
import numpy as np

# Converter RGB para HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# Detectar objetos vermelhos
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# Para vermelho também pode haver transição no limite do círculo cromático
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

# Combinar máscaras
red_mask = mask1 + mask2
```

### Outros Espaços de Cor
- **LAB**: Separar luminância de cor (L*a*b*)
- **YUV/YCrCb**: Separar luminância de crominância
- **Grayscale**: Conversão para tons de cinza

## Histogramas

O histograma de uma imagem mostra a distribuição de intensidades de pixel, útil para análise e correção de contraste.

### Histograma em Escala de Cinza

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Calcular histograma
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plotar histograma
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Imagem em Escala de Cinza')
plt.subplot(1, 2, 2)
plt.plot(histogram)
plt.title('Histograma')
plt.xlabel('Intensidade')
plt.ylabel('Número de Pixels')
plt.show()
```

### Equalização de Histograma

Melhora o contraste de imagens com baixo contraste:

```python
# Equalização de histograma
equalized = cv2.equalizeHist(gray)

# Comparar antes e depois
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].plot(cv2.calcHist([gray], [0], None, [256], [0, 256]))
axes[0,1].set_title('Histograma Original')

axes[1,0].imshow(equalized, cmap='gray')
axes[1,0].set_title('Equalizada')
axes[1,1].plot(cv2.calcHist([equalized], [0], None, [256], [0, 256]))
axes[1,1].set_title('Histograma Equalizado')
plt.tight_layout()
plt.show()
```

## Filtros e Convolução

A convolução é uma operação fundamental em processamento de imagem, usada para aplicar filtros e transformações locais.

### Tipos de Filtros

#### Filtros Passa-Baixa (Suavização)
- Reduzem ruído e detalhes finos
- Média, gaussiano, mediana

```python
# Filtro de média
blurred_avg = cv2.blur(gray, (5, 5))

# Filtro Gaussiano
blurred_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# Filtro de mediana (bom para ruído sal e pimenta)
blurred_median = cv2.medianBlur(gray, 5)

# Comparar resultados
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(blurred_avg, cmap='gray')
axes[0,1].set_title('Filtro Média')
axes[1,0].imshow(blurred_gaussian, cmap='gray')
axes[1,0].set_title('Filtro Gaussiano')
axes[1,1].imshow(blurred_median, cmap='gray')
axes[1,1].set_title('Filtro Mediana')
plt.tight_layout()
plt.show()
```

#### Filtros Passa-Alta (Realce de Bordas)
- Destacam variações abruptas de intensidade
- Laplaciano, Sobel, Canny

```python
# Detector de bordas Sobel
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

# Detector de bordas Laplaciano
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Detector de bordas Canny
canny = cv2.Canny(gray, 100, 200)

# Comparar resultados
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(np.absolute(sobel_combined), cmap='gray')
axes[0,1].set_title('Sobel')
axes[1,0].imshow(np.absolute(laplacian), cmap='gray')
axes[1,0].set_title('Laplaciano')
axes[1,1].imshow(canny, cmap='gray')
axes[1,1].set_title('Canny')
plt.tight_layout()
plt.show()
```

## Binarização e Segmentação

### Binarização Global
Converte imagem em escala de cinza para preto e branco com base em um limiar (threshold):

```python
# Binarização global
_, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Binarização adaptativa
binary_adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# Otsu's binarization
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Comparar métodos
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(binary_thresh, cmap='gray')
axes[0,1].set_title('Limiar Fixo')
axes[1,0].imshow(binary_adaptive, cmap='gray')
axes[1,0].set_title('Adaptativo')
axes[1,1].imshow(binary_otsu, cmap='gray')
axes[1,1].set_title('Otsu')
plt.tight_layout()
plt.show()
```

## Operações Morfológicas

Operações morfológicas manipulam a estrutura geométrica dos objetos em uma imagem binária:

```python
# Elemento estruturante
kernel = np.ones((5,5), np.uint8)

# Erosão: reduz objetos brancos
erosion = cv2.erode(binary_thresh, kernel, iterations=1)

# Dilatação: aumenta objetos brancos
dilation = cv2.dilate(binary_thresh, kernel, iterations=1)

# Abertura: erosão seguida de dilatação (remove ruído)
opening = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, kernel)

# Fechamento: dilatação seguida de erosão (fecha lacunas)
closing = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)

# Comparar operações
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(binary_thresh, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(erosion, cmap='gray')
axes[0,1].set_title('Erosão')
axes[0,2].imshow(dilation, cmap='gray')
axes[0,2].set_title('Dilatação')
axes[1,0].imshow(opening, cmap='gray')
axes[1,0].set_title('Abertura')
axes[1,1].imshow(closing, cmap='gray')
axes[1,1].set_title('Fechamento')
axes[1,2].axis('off')
plt.tight_layout()
plt.show()
```

## Contornos e Bounding Boxes

Contornos são curvas que conectam pontos contínuos de mesma intensidade, úteis para detecção de formas e segmentação:

```python
import cv2
import numpy as np

# Encontrar contornos
contours, hierarchy = cv2.findContours(
    binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Desenhar contornos
img_with_contours = img_rgb.copy()
cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

# Calcular bounding boxes
img_bboxes = img_rgb.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_bboxes, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Calcular bounding boxes rotacionadas
img_rotated_bboxes = img_rgb.copy()
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_rotated_bboxes, [box], 0, (0, 255, 255), 2)

# Visualizar resultados
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[1].imshow(img_with_contours)
axes[1].set_title('Contornos')
axes[2].imshow(img_bboxes)
axes[2].set_title('Bounding Boxes')
plt.tight_layout()
plt.show()

print(f"Número de contornos encontrados: {len(contours)}")
```

## Técnicas Avançadas de Segmentação

### Segmentação por Crescimento de Região (Region Growing)
Agrupa pixels com características semelhantes:

```python
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float

# Segmentação de Felzenszwalb
segments = felzenszwalb(img_as_float(img_rgb), scale=100, sigma=0.5, min_size=50)

# Visualizar segmentos
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(segments, cmap='nipy_spectral')
plt.title('Segmentos')
plt.show()
```

### Segmentação por Watershed
Técnica baseada em morfologia matemática:

```python
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Converter para escala de cinza
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Calcular distância euclidiana
distance = ndimage.distance_transform_edt(binary_thresh)

# Encontrar picos locais
local_maxima = peak_local_max(distance, min_distance=20, labels=binary_thresh)

# Criar marcadores
markers, _ = ndimage.label(local_maxima)
segmented = watershed(-distance, markers, mask=binary_thresh)

# Visualizar resultado
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(distance, cmap='hot')
plt.title('Distância')
plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap='nipy_spectral')
plt.title('Watershed')
plt.show()
```

## Aplicações Práticas

### Detecção de Objetos por Cor
```python
def detect_objects_by_color(image, lower_color, upper_color):
    """Detecta objetos de uma determinada cor em uma imagem"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Criar máscara para a cor
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Aplicar operações morfológicas para limpar a máscara
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar bounding boxes
    result = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filtrar pequenos contornos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return result, len(contours)

# Exemplo: detectar objetos vermelhos
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
result_img, count = detect_objects_by_color(img_rgb, lower_red, upper_red)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(result_img)
plt.title(f'Detecção de Objetos Vermelhos ({count} encontrados)')
plt.show()
```

## Considerações Práticas

### Escolha de Técnicas
- **Iluminação**: Use espaços de cor adequados (HSV para variações de brilho)
- **Ruído**: Aplique filtros antes de operações sensíveis
- **Desempenho**: Considere complexidade computacional para aplicações em tempo real
- **Precisão**: Ajuste parâmetros com base nas características específicas do problema

### Avaliação de Resultados
- **Precisão**: Proporção de pixels corretamente classificados
- **Recall**: Proporção de pixels relevantes detectados
- **IoU (Intersection over Union)**: Medida de sobreposição entre segmentação predita e real