# Unidade 3 - Extração de Características e Machine Learning

## Introdução à Extração de Características

A extração de características é um passo crítico em sistemas de visão computacional, onde informações relevantes são extraídas de imagens para serem usadas em tarefas de classificação, detecção ou reconhecimento. Uma boa representação de características permite que algoritmos de aprendizado de máquina identifiquem padrões de forma eficiente.

### O que são Características (Features)?

Características são propriedades quantificáveis de uma imagem que representam aspectos importantes para uma tarefa específica. Podem ser:

- **Características de baixo nível**: Bordas, texturas, cores, gradientes
- **Características de médio nível**: Cantos, blobs, keypoints
- **Características de alto nível**: Objetos, partes de objetos, relações espaciais

## Keypoints e Descritores

### Keypoints (Pontos de Interesse)

Keypoints são pontos específicos em uma imagem que possuem propriedades distintivas, como cantos, bordas ou regiões com alta variação de intensidade. São invariantes a transformações como rotação, escala e iluminação.

### Descritores

Descritores são vetores numéricos que codificam informações sobre a vizinhança de um keypoint, permitindo comparação entre diferentes keypoints.

### Algoritmos Populares

#### Harris Corner Detector
Detecta cantos em imagens com base na variação de intensidade em diferentes direções:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_harris_corners(image, blockSize=2, ksize=3, k=0.04):
    """Detecta cantos usando o detector de Harris"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Calcular derivadas
    dx2 = cv2.cornerHarris(gray, blockSize, ksize, k)
    
    # Dilatar para marcar cantos claramente
    dx2 = cv2.dilate(dx2, None)
    
    # Retornar imagem com cantos destacados
    result = image.copy()
    result[dx2 > 0.01*dx2.max()] = [255, 0, 0]  # Marcar cantos em vermelho
    
    return result, dx2

# Exemplo de uso
img = cv2.imread('exemplo.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

corners_img, corners_map = detect_harris_corners(img_rgb)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Imagem Original')
plt.subplot(1, 2, 2)
plt.imshow(corners_img)
plt.title('Cantos Detectados (Harris)')
plt.show()
```

#### Shi-Tomasi Corner Detector
Melhoria do detector de Harris, selecionando cantos com maior precisão:

```python
def detect_shi_tomasi_corners(image, maxCorners=100, qualityLevel=0.01, minDistance=10):
    """Detecta cantos usando o detector Shi-Tomasi"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners, qualityLevel, minDistance
    )
    
    # Converter para int
    corners = np.int0(corners)
    
    # Desenhar cantos
    result = image.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(result, (x, y), 5, [255, 0, 0], -1)
    
    return result, corners

# Exemplo de uso
shi_tomasi_img, shi_tomasi_corners = detect_shi_tomasi_corners(img_rgb)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Imagem Original')
plt.subplot(1, 2, 2)
plt.imshow(shi_tomasi_img)
plt.title(f'Cantos Detectados (Shi-Tomasi) - {len(shi_tomasi_corners)} encontrados')
plt.show()
```

#### SIFT (Scale-Invariant Feature Transform)
Algoritmo robusto para detecção e descrição de keypoints invariantes a escala e rotação:

```python
def detect_sift_features(image):
    """Detecta e descreve keypoints usando SIFT"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Criar detector SIFT
    sift = cv2.SIFT_create()
    
    # Detectar keypoints e calcular descritores
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Desenhar keypoints
    result = cv2.drawKeypoints(image, keypoints, None)
    
    return result, keypoints, descriptors

# Exemplo de uso (se o OpenCV tiver SIFT disponível)
try:
    sift_img, sift_kp, sift_desc = detect_sift_features(img_rgb)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Imagem Original')
    plt.subplot(1, 2, 2)
    plt.imshow(sift_img)
    plt.title(f'SIFT Features - {len(sift_kp) if sift_kp else 0} keypoints')
    plt.show()
except cv2.error:
    print("SIFT não disponível nesta versão do OpenCV")
```

#### ORB (Oriented FAST and Rotated BRIEF)
Versão mais rápida e livre de patentes do SIFT:

```python
def detect_orb_features(image):
    """Detecta e descreve keypoints usando ORB"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Criar detector ORB
    orb = cv2.ORB_create()
    
    # Detectar keypoints e calcular descritores
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Desenhar keypoints
    result = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)
    
    return result, keypoints, descriptors

# Exemplo de uso
orb_img, orb_kp, orb_desc = detect_orb_features(img_rgb)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Imagem Original')
plt.subplot(1, 2, 2)
plt.imshow(orb_img)
plt.title(f'ORB Features - {len(orb_kp) if orb_kp else 0} keypoints')
plt.show()
```

## Vetorização de Imagens

Converter imagens em vetores numéricos é essencial para aplicar algoritmos de machine learning. Existem várias abordagens para isso:

### Vetorização Simples
Converter pixels diretamente em um vetor:

```python
def flatten_image(image):
    """Converte imagem em vetor 1D"""
    if len(image.shape) == 3:  # RGB
        flattened = image.reshape(-1)
    else:  # Grayscale
        flattened = image.flatten()
    return flattened

# Exemplo
sample_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
flattened_vector = flatten_image(sample_img)
print(f"Tamanho original: {sample_img.shape}")
print(f"Tamanho vetorializado: {flattened_vector.shape}")
```

### Histogramas como Vetores de Características
```python
def compute_color_histogram(image, bins=32):
    """Calcula histograma de cores como vetor de características"""
    # Converter para HSV para melhor representação de cor
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calcular histograma para cada canal
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    
    # Concatenar histogramas
    feature_vector = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    return feature_vector

# Exemplo
hist_features = compute_color_histogram(img_rgb)
print(f"Vetor de características (histograma): {hist_features.shape}")
```

### Descritores de Textura
```python
from skimage.feature import greycomatrix, greycoprops

def compute_texture_features(image):
    """Calcula descritores de textura GLCM"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calcular matriz de co-ocorrência de níveis de cinza
    glcm = greycomatrix(gray, [1], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)
    
    # Extrair propriedades
    contrast = greycoprops(glcm, 'contrast').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    correlation = greycoprops(glcm, 'correlation').flatten()
    
    # Concatenar todas as propriedades
    texture_features = np.concatenate([contrast, energy, homogeneity, correlation])
    
    return texture_features

# Exemplo
texture_features = compute_texture_features(img_rgb)
print(f"Vetor de características (textura): {texture_features.shape}")
```

## Classificação Supervisionada

### Pipeline de Classificação

1. **Extração de características**: Converter imagens em vetores
2. **Divisão dos dados**: Treino, validação e teste
3. **Treinamento do modelo**: Ajustar parâmetros
4. **Avaliação**: Medir desempenho
5. **Otimização**: Ajustar hiperparâmetros

### Exemplo Completo de Classificação

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import os

class ImageClassifier:
    def __init__(self, classifier_type='random_forest'):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', random_state=42)
        
    def extract_features(self, image):
        """Extrai características de uma única imagem"""
        # Histograma de cores
        color_hist = compute_color_histogram(image)
        
        # Características de textura
        texture_features = compute_texture_features(image)
        
        # Concatenar todas as características
        features = np.concatenate([color_hist, texture_features])
        
        return features
    
    def prepare_dataset(self, image_paths, labels):
        """Prepara o dataset completo"""
        features_list = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features = self.extract_features(img)
            features_list.append(features)
        
        X = np.array(features_list)
        y = np.array(labels)
        
        return X, y
    
    def train(self, X_train, y_train):
        """Treina o classificador"""
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Treinar modelo
        self.classifier.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        """Faz previsões"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.classifier.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return accuracy, report

# Exemplo de uso (com dados hipotéticos)
# Suponha que temos caminhos de imagens e rótulos
# image_paths = ['img1.jpg', 'img2.jpg', ...]
# labels = ['classe_a', 'classe_b', ...]

# classifier = ImageClassifier(classifier_type='random_forest')
# X, y = classifier.prepare_dataset(image_paths, labels)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# classifier.train(X_train, y_train)
# accuracy, report = classifier.evaluate(X_test, y_test)
# print(f"Acurácia: {accuracy}")
# print(report)
```

## Avaliação de Modelos

### Métricas Comuns

```python
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()
    
    # Calcular métricas
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f"Precisão: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

# Exemplo de uso
# plot_confusion_matrix(y_test, predictions, classes=['Classe A', 'Classe B', 'Classe C'])
```

### Curva ROC e AUC
Para problemas binários:

```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curves(y_true, y_scores, n_classes):
    """Plota curvas ROC para classificação multiclasse"""
    # Binarizar rótulos
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calcular curvas ROC para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plotar curvas
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC')
    plt.legend(loc="lower right")
    plt.show()
```

## Técnicas Avançadas de Extração de Características

### Bag of Visual Words (BOVW)
Similar ao Bag of Words em processamento de linguagem natural:

```python
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

class BagOfVisualWords:
    def __init__(self, vocab_size=100, descriptor_extractor=None):
        self.vocab_size = vocab_size
        self.kmeans = KMeans(n_clusters=vocab_size, random_state=42)
        self.descriptor_extractor = descriptor_extractor or self.default_descriptor_extractor
        
    def default_descriptor_extractor(self, image):
        """Extrator de descritores padrão (ORB)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return descriptors
    
    def build_vocabulary(self, images):
        """Constrói vocabulário a partir de descritores de múltiplas imagens"""
        all_descriptors = []
        
        for img in images:
            descriptors = self.descriptor_extractor(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        # Concatenar todos os descritores
        all_descriptors = np.vstack(all_descriptors)
        
        # Treinar K-means para formar vocabulário
        self.kmeans.fit(all_descriptors)
        
    def encode_image(self, image):
        """Codifica uma imagem como histograma de palavras visuais"""
        descriptors = self.descriptor_extractor(image)
        
        if descriptors is None:
            return np.zeros(self.vocab_size)
        
        # Atribuir descritores aos clusters mais próximos
        distances = cdist(descriptors, self.kmeans.cluster_centers_)
        assignments = np.argmin(distances, axis=1)
        
        # Criar histograma
        histogram = np.bincount(assignments, minlength=self.vocab_size)
        
        # Normalizar
        histogram = histogram.astype(float) / histogram.sum()
        
        return histogram
    
    def encode_dataset(self, images):
        """Codifica todas as imagens do dataset"""
        histograms = []
        
        for img in images:
            hist = self.encode_image(img)
            histograms.append(hist)
        
        return np.array(histograms)

# Exemplo de uso
# bovw = BagOfVisualWords(vocab_size=200)
# bovw.build_vocabulary(training_images)
# train_histograms = bovw.encode_dataset(training_images)
# test_histograms = bovw.encode_dataset(test_images)
```

### HOG (Histogram of Oriented Gradients)
Eficiente para detecção de objetos:

```python
from skimage.feature import hog
from skimage import exposure

def extract_hog_features(image, visualize=False):
    """Extrai características HOG de uma imagem"""
    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calcular HOG
    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    
    if visualize:
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(gray, cmap='gray')
        ax1.set_title('Imagem Original')
        ax1.axis('off')
        
        ax2.imshow(hog_image_rescaled, cmap='gray')
        ax2.set_title('HOG Features')
        ax2.axis('off')
        
        plt.show()
    
    return features

# Exemplo
hog_features = extract_hog_features(img_rgb, visualize=True)
print(f"Vetor de características HOG: {hog_features.shape}")
```

## Considerações Práticas

### Seleção de Características
- **Relevância**: Escolher características que realmente contribuam para a tarefa
- **Redundância**: Evitar características altamente correlacionadas
- **Dimensionalidade**: Equilibrar quantidade de características com desempenho

### Preprocessamento
- **Normalização**: Padronizar características para evitar domínio de algumas sobre outras
- **Redução de dimensionalidade**: Usar PCA ou LDA para simplificar o espaço de características

### Validação Cruzada
Importante para estimar o desempenho real do modelo:

```python
from sklearn.model_selection import cross_val_score

def evaluate_model_cv(model, X, y, cv=5):
    """Avalia modelo com validação cruzada"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"Acurácias na validação cruzada: {scores}")
    print(f"Média: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    return scores

# Exemplo
# evaluate_model_cv(RandomForestClassifier(), X, y)
```

## Overfitting e Regularização

O overfitting ocorre quando o modelo aprende demais os dados de treinamento, perdendo capacidade de generalização:

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """Plota curva de aprendizado para diagnosticar overfitting"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Treino')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validação')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.title(title)
    plt.xlabel('Tamanho do Conjunto de Treinamento')
    plt.ylabel('Acurácia')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Exemplo
# plot_learning_curve(RandomForestClassifier(), X, y)
```

Essas técnicas de extração de características e classificação supervisionada formam a base para muitas aplicações de visão computacional, especialmente quando não há dados suficientes para usar deep learning ou quando é necessário um modelo mais interpretável.