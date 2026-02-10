# Aula 6 - Classificação com ML Tradicional

## Objetivo da Aula

Implementar classificadores baseados em machine learning tradicional, utilizando extração de features e pipelines sklearn, com foco em validação e prevenção de overfitting.

## Conteúdo Teórico

### Extração de Features para Classificação

Para classificação tradicional de imagens, é necessário converter imagens em vetores numéricos que representem características relevantes:

- **Features de cor**: Histogramas de cores em diferentes espaços (RGB, HSV, LAB)
- **Features de textura**: Propriedades estatísticas da distribuição de intensidades
- **Features de forma**: Características geométricas dos objetos
- **Features compostas**: Combinações de diferentes tipos de features

### Pipeline Sklearn

O scikit-learn oferece uma estrutura padronizada para construção de pipelines de ML:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
```

### Técnicas de Validação

- **Holdout**: Divisão simples em treino/teste
- **Cross-validation**: Validação cruzada para estimativa mais robusta
- **Stratified sampling**: Manutenção da proporção de classes em divisões

### Overfitting

Ocorre quando o modelo aprende excessivamente os dados de treinamento, perdendo capacidade de generalização. Técnicas para mitigação:

- **Regularização**: Penalização de modelos complexos
- **Validação cruzada**: Estimativa mais realista do desempenho
- **Early stopping**: Parada prematura do treinamento
- **Dados adicionais**: Aumento do conjunto de treinamento

## Atividade Prática

### Implementar Extrator de Features

```python
# src/features/traditional_features.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TraditionalImageFeatures:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def extract_color_histogram(self, image, bins=32):
        """Extrai histograma de cores como feature"""
        # Converter para HSV para melhor representação de cor
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calcular histograma para cada canal
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalizar histogramas
        hist_h = hist_h.flatten() / hist_h.sum()
        hist_s = hist_s.flatten() / hist_s.sum()
        hist_v = hist_v.flatten() / hist_v.sum()
        
        # Concatenar histogramas
        feature_vector = np.concatenate([hist_h, hist_s, hist_v])
        
        return feature_vector
    
    def extract_texture_features(self, image):
        """Extrai features de textura usando LBP e estatísticas"""
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Histograma do LBP
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        
        # Estatísticas básicas
        mean = np.mean(gray)
        std = np.std(gray)
        skewness = np.mean(((gray - mean) / std) ** 3) if std != 0 else 0
        kurtosis = np.mean(((gray - mean) / std) ** 4) if std != 0 else 3
        
        # Concatenar features
        texture_features = np.concatenate([lbp_hist, [mean, std, skewness, kurtosis]])
        
        return texture_features
    
    def extract_shape_features(self, image):
        """Extrai features de forma baseadas em contornos"""
        # Converter para escala de cinza e binarizar
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Retornar zeros se não houver contornos
            return np.zeros(10)
        
        # Pegar o maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular features de forma
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularidade
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Razão de aspecto
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Extensão (extent)
        extent = float(area) / (w * h) if w * h != 0 else 0
        
        # Solidez (solidity)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Equivalent diameter
        equiv_diameter = np.sqrt(4 * area / np.pi) if area >= 0 else 0
        
        # Contar buracos (componentes internos)
        # Esta é uma aproximação simples
        holes = len(cv2.findContours(255 - binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]) if area > 0 else 0
        
        shape_features = np.array([
            area, perimeter, circularity, aspect_ratio, 
            extent, solidity, equiv_diameter, holes, w, h
        ])
        
        return shape_features
    
    def extract_hog_features(self, image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """Extrai features HOG (Histogram of Oriented Gradients)"""
        # Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calcular HOG
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )
        
        return features
    
    def extract_all_features(self, image):
        """Extrai todas as features e concatena"""
        color_features = self.extract_color_histogram(image)
        texture_features = self.extract_texture_features(image)
        shape_features = self.extract_shape_features(image)
        hog_features = self.extract_hog_features(image)
        
        # Concatenar todas as features
        all_features = np.concatenate([
            color_features,
            texture_features,
            shape_features,
            hog_features
        ])
        
        return all_features
    
    def extract_features_batch(self, images):
        """Extrai features para um batch de imagens"""
        features_list = []
        
        for img in images:
            features = self.extract_all_features(img)
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit_scaler(self, features):
        """Ajusta o scaler com base nas features"""
        self.scaler.fit(features)
    
    def transform_features(self, features):
        """Transforma features usando o scaler ajustado"""
        return self.scaler.transform(features)
```

### Implementar Pipeline de Classificação

```python
# src/models/traditional_classifier.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class TraditionalImageClassifier:
    def __init__(self, classifier_type='random_forest', **kwargs):
        """
        Inicializa o classificador tradicional
        classifier_type: 'random_forest', 'svm', 'logistic_regression'
        """
        self.classifier_type = classifier_type
        self.feature_extractor = None
        self.pipeline = None
        self.classes_ = None
        
        # Configurar classificador baseado no tipo
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(random_state=42, **kwargs)
        elif classifier_type == 'svm':
            self.classifier = SVC(random_state=42, **kwargs)
        elif classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(random_state=42, **kwargs)
        else:
            raise ValueError(f"Tipo de classificador não suportado: {classifier_type}")
    
    def setup_pipeline(self):
        """Configura o pipeline com scaler e classificador"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.classifier)
        ])
    
    def prepare_data(self, images, labels, feature_extractor):
        """Prepara dados para treinamento"""
        self.feature_extractor = feature_extractor
        
        # Extrair features
        X = feature_extractor.extract_features_batch(images)
        
        # Armazenar classes
        self.classes_ = np.unique(labels)
        
        return X, labels
    
    def train(self, X, y, cv_folds=5):
        """Treina o classificador"""
        if self.pipeline is None:
            self.setup_pipeline()
        
        # Treinar o pipeline
        self.pipeline.fit(X, y)
        
        # Validar com cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv_folds, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores
    
    def predict(self, X):
        """Faz previsões"""
        if self.pipeline is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Faz previsões com probabilidades (se suportado)"""
        if self.pipeline is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Verificar se o classificador suporta probabilidades
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise NotImplementedError("Este classificador não suporta previsão de probabilidades")
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        if self.pipeline is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Fazer previsões
        y_pred = self.predict(X_test)
        
        # Calcular acurácia
        accuracy = accuracy_score(y_test, y_pred)
        
        # Gerar relatório de classificação
        report = classification_report(y_test, y_pred, target_names=self.classes_)
        
        # Gerar matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Acurácia no conjunto de teste: {accuracy:.3f}")
        print("\nRelatório de Classificação:")
        print(report)
        
        return accuracy, report, cm
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Matriz de Confusão"):
        """Plota matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title(title)
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.show()
    
    def get_feature_importance(self):
        """Obtém importância das features (se disponível)"""
        if self.classifier_type == 'random_forest':
            return self.pipeline.named_steps['classifier'].feature_importances_
        else:
            raise NotImplementedError("Importância de features disponível apenas para Random Forest")
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        model_data = {
            'pipeline': self.pipeline,
            'classes': self.classes_,
            'classifier_type': self.classifier_type
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo previamente treinado"""
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.classes_ = model_data['classes']
        self.classifier_type = model_data['classifier_type']
        print(f"Modelo carregado de: {filepath}")
```

### Implementar Pipeline Completo

```python
# src/pipelines/traditional_ml_pipeline.py
from features.traditional_features import TraditionalImageFeatures
from models.traditional_classifier import TraditionalImageClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class TraditionalMLPipeline:
    def __init__(self, classifier_type='random_forest', feature_params=None):
        self.feature_extractor = TraditionalImageFeatures()
        self.classifier = TraditionalImageClassifier(classifier_type)
        self.feature_params = feature_params or {}
        self.is_trained = False
    
    def prepare_dataset(self, images, labels):
        """Prepara o dataset completo"""
        # Extrair features
        print("Extraindo features...")
        X = self.feature_extractor.extract_features_batch(images)
        y = np.array(labels)
        
        print(f"Features extraídas: {X.shape}")
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Conjunto de treino: {X_train.shape[0]} amostras")
        print(f"Conjunto de teste: {X_test.shape[0]} amostras")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, cv_folds=5):
        """Treina o modelo"""
        print("Treinando modelo...")
        
        # Treinar o classificador
        cv_scores = self.classifier.train(X_train, y_train, cv_folds)
        
        self.is_trained = True
        
        return cv_scores
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")
        
        print("Avaliando modelo...")
        accuracy, report, cm = self.classifier.evaluate(X_test, y_test)
        
        return accuracy, report, cm
    
    def predict_single(self, image):
        """Faz previsão para uma única imagem"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Extrair features da imagem
        features = self.feature_extractor.extract_all_features(image)
        features = features.reshape(1, -1)  # Adicionar dimensão de batch
        
        # Fazer previsão
        prediction = self.classifier.predict(features)[0]
        
        # Obter probabilidade se disponível
        try:
            probabilities = self.classifier.predict_proba(features)[0]
            return prediction, probabilities
        except:
            return prediction, None
    
    def get_model_info(self):
        """Obtém informações sobre o modelo"""
        if not self.is_trained:
            return "Modelo não treinado"
        
        info = {
            'classifier_type': self.classifier.classifier_type,
            'feature_dimension': self.feature_extractor.extract_all_features(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            ).shape[0],
            'classes': self.classifier.classes_.tolist() if self.classifier.classes_ is not None else None
        }
        
        return info
```

### Exemplo de Uso

```python
# src/examples/traditional_ml_example.py
from pipelines.traditional_ml_pipeline import TraditionalMLPipeline
from utils.io import load_image_rgb
import numpy as np
import matplotlib.pyplot as plt

def create_sample_dataset():
    """Cria um dataset de exemplo (substituir com dados reais)"""
    # Este é um exemplo simplificado
    # Em uma aplicação real, você carregaria imagens reais
    
    # Simular imagens de 3 classes diferentes
    n_samples_per_class = 50
    classes = ['classe_a', 'classe_b', 'classe_c']
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        for _ in range(n_samples_per_class):
            # Criar imagem simulada com padrão diferente para cada classe
            if class_name == 'classe_a':
                # Padrão com mais vermelho
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img[:, :, 0] = np.clip(img[:, :, 0] + 50, 0, 255)  # Mais vermelho
            elif class_name == 'classe_b':
                # Padrão com mais verde
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img[:, :, 1] = np.clip(img[:, :, 1] + 50, 0, 255)  # Mais verde
            else:  # classe_c
                # Padrão com mais azul
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img[:, :, 2] = np.clip(img[:, :, 2] + 50, 0, 255)  # Mais azul
            
            images.append(img)
            labels.append(class_name)
    
    return images, labels

def demonstrate_traditional_ml():
    """Demonstra classificação com ML tradicional"""
    print("=== Demonstração de Classificação com ML Tradicional ===\n")
    
    # Criar dataset de exemplo
    print("Criando dataset de exemplo...")
    images, labels = create_sample_dataset()
    
    print(f"Dataset criado: {len(images)} imagens, {len(set(labels))} classes")
    
    # Inicializar pipeline
    print("\nInicializando pipeline...")
    pipeline = TraditionalMLPipeline(classifier_type='random_forest')
    
    # Preparar dataset
    X_train, X_test, y_train, y_test = pipeline.prepare_dataset(images, labels)
    
    # Treinar modelo
    cv_scores = pipeline.train(X_train, y_train, cv_folds=5)
    
    # Avaliar modelo
    accuracy, report, cm = pipeline.evaluate(X_test, y_test)
    
    # Informações do modelo
    model_info = pipeline.get_model_info()
    print(f"\nInformações do modelo: {model_info}")
    
    # Testar previsão em uma imagem
    print(f"\nTestando previsão em uma imagem...")
    sample_prediction, sample_probabilities = pipeline.predict_single(images[0])
    print(f"Predição para imagem de exemplo: {sample_prediction}")
    if sample_probabilities is not None:
        print(f"Probabilidades: {sample_probabilities}")
    
    # Plotar matriz de confusão
    pipeline.classifier.plot_confusion_matrix(y_test, pipeline.classifier.predict(X_test))
    
    return pipeline

def compare_classifiers():
    """Compara diferentes classificadores tradicionais"""
    print("\n=== Comparação de Classificadores Tradicionais ===\n")
    
    # Criar dataset
    images, labels = create_sample_dataset()
    
    classifiers = ['random_forest', 'svm', 'logistic_regression']
    results = {}
    
    for clf_type in classifiers:
        print(f"\nTreinando {clf_type}...")
        
        pipeline = TraditionalMLPipeline(classifier_type=clf_type)
        X_train, X_test, y_train, y_test = pipeline.prepare_dataset(images, labels)
        
        # Treinar
        pipeline.train(X_train, y_train, cv_folds=3)  # Menos folds para acelerar
        
        # Avaliar
        accuracy, _, _ = pipeline.evaluate(X_test, y_test)
        results[clf_type] = accuracy
        
        print(f"Acurácia de {clf_type}: {accuracy:.3f}")
    
    print(f"\nResultados finais:")
    for clf_type, acc in results.items():
        print(f"  {clf_type}: {acc:.3f}")
    
    best_clf = max(results, key=results.get)
    print(f"\nMelhor classificador: {best_clf} com acurácia {results[best_clf]:.3f}")
    
    return results

def analyze_overfitting():
    """Analisa potencial overfitting"""
    print("\n=== Análise de Overfitting ===\n")
    
    # Criar dataset com mais amostras para análise
    images, labels = create_sample_dataset()
    
    # Dividir em treino e teste
    from sklearn.model_selection import train_test_split
    X_temp, _, y_temp, _ = train_test_split(
        np.array(images), np.array(labels), test_size=0.2, random_state=42, stratify=labels
    )
    
    # Criar versões com diferentes tamanhos
    sizes = [10, 20, 30, 40, 50]
    train_accuracies = []
    val_accuracies = []
    
    for size in sizes:
        # Pegar subset dos dados
        subset_indices = np.random.choice(len(X_temp), size=size, replace=False)
        X_subset = [images[i] for i in subset_indices]
        y_subset = [labels[i] for i in subset_indices]
        
        pipeline = TraditionalMLPipeline(classifier_type='random_forest')
        X_train, X_test, y_train, y_test = pipeline.prepare_dataset(X_subset, y_subset)
        
        # Treinar
        pipeline.train(X_train, y_train, cv_folds=min(3, len(np.unique(y_train))))
        
        # Avaliar em treino e validação
        train_pred = pipeline.classifier.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        val_pred = pipeline.classifier.predict(X_test)
        val_acc = np.mean(val_pred == y_test)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"Tamanho: {size}, Treino: {train_acc:.3f}, Validação: {val_acc:.3f}")
    
    # Plotar curvas de aprendizado
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_accuracies, 'o-', label='Treino', color='blue')
    plt.plot(sizes, val_accuracies, 'o-', label='Validação', color='red')
    plt.xlabel('Tamanho do Conjunto de Treino')
    plt.ylabel('Acurácia')
    plt.title('Curva de Aprendizado - Análise de Overfitting')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Executar demonstrações
    trained_pipeline = demonstrate_traditional_ml()
    compare_classifiers()
    analyze_overfitting()
```

## Resultado Esperado

Nesta aula, você:

1. Implementou extratores de features tradicionais (cor, textura, forma, HOG)
2. Criou um pipeline completo de classificação com sklearn
3. Comparou diferentes algoritmos de ML tradicional
4. Aplicou técnicas de validação cruzada
5. Analisou o fenômeno de overfitting
6. Testou o sistema com diferentes configurações

Este pipeline de ML tradicional serve como base importante para compreensão dos fundamentos de classificação de imagens, mesmo com o advento do deep learning.