# Aula 9 - CNN na Prática (Sem Matemática Excessiva)

## Objetivo da Aula

Compreender o funcionamento interno das Redes Neurais Convolucionais (CNNs) de forma visual e intuitiva, implementar transfer learning com modelos pré-existentes e aplicar fine-tuning em datasets específicos.

## Conteúdo Teórico

### Como Funcionam as CNNs (Visão Intuitiva)

As Redes Neurais Convolucionais são projetadas para reconhecer padrões visuais em imagens. Podemos pensar nelas como um sistema com múltiplas etapas de "detecção":

1. **Camadas Iniciais**: Detectam padrões simples como linhas, bordas e cantos
2. **Camadas Intermediárias**: Combinam padrões simples para formar formas mais complexas
3. **Camadas Finais**: Reconhecem objetos completos com base nos padrões anteriores

### Transfer Learning

O Transfer Learning é uma técnica que permite reutilizar um modelo pré-treinado em uma nova tarefa, economizando tempo e recursos computacionais.

**Vantagens:**
- Menos dados necessários
- Tempo de treinamento reduzido
- Melhor desempenho em datasets menores
- Menor poder computacional exigido

### Arquiteturas Populares

- **VGG**: Simples e eficaz, bom para aprendizado
- **ResNet**: Excelente desempenho, permite redes muito profundas
- **MobileNet**: Otimizado para dispositivos móveis
- **EfficientNet**: Bom equilíbrio entre acurácia e eficiência

## Atividade Prática

### Implementar Transfer Learning com TensorFlow/Keras

```python
# src/models/cnn_transfer_learning.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

class CNNTransferLearning:
    def __init__(self, base_model_name='vgg16', input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.model = None
        self.history = None
        
        # Carregar modelo base
        self.base_model = self._load_base_model()
    
    def _load_base_model(self):
        """Carrega modelo pré-treinado"""
        if self.base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name.lower() == 'mobilenetv2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Modelo {self.base_model_name} não suportado")
        
        # Congelar pesos do modelo base
        base_model.trainable = False
        
        return base_model
    
    def build_model(self):
        """Constrói modelo com cabeça personalizada"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Pré-processamento do modelo base
        if self.base_model_name.lower() in ['vgg16', 'resnet50']:
            x = tf.keras.applications.vgg16.preprocess_input(inputs)
        elif self.base_model_name.lower() == 'mobilenetv2':
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Extrair features
        x = self.base_model(x, training=False)
        
        # Camadas personalizadas
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compila o modelo"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_initial(self, train_dataset, validation_dataset, epochs=10):
        """Treina a cabeça personalizada"""
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            verbose=1
        )
        return self.history
    
    def unfreeze_model(self, fine_tune_from_layer=100):
        """Descongela parte do modelo base para fine-tuning"""
        self.base_model.trainable = True
        
        # Congelar as primeiras camadas (menos específicas)
        fine_tune_at = len(self.base_model.layers) - fine_tune_from_layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    def fine_tune(self, train_dataset, validation_dataset, epochs=10, learning_rate=0.0001/10):
        """Realiza fine-tuning"""
        # Recompilar com learning rate menor
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Treinar novamente
        fine_tune_history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            verbose=1
        )
        
        return fine_tune_history
    
    def visualize_model_architecture(self):
        """Visualiza arquitetura do modelo"""
        if self.model:
            print(self.model.summary())
        else:
            print("Modelo não construído ainda")
    
    def plot_training_history(self):
        """Plota histórico de treinamento"""
        if self.history:
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']
            
            epochs_range = range(len(acc))
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Acurácia de Treino')
            plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
            plt.legend(loc='lower right')
            plt.title('Acurácia de Treino e Validação')
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Perda de Treino')
            plt.plot(epochs_range, val_loss, label='Perda de Validação')
            plt.legend(loc='upper right')
            plt.title('Perda de Treino e Validação')
            
            plt.show()
        else:
            print("Nenhum histórico de treinamento disponível")
```

### Visualização de Features Extraídas

```python
# src/utils/cnn_visualization.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_maps(model, image, layer_names=None):
    """Visualiza feature maps de camadas específicas"""
    # Preparar imagem
    img = tf.expand_dims(image, axis=0)
    
    # Criar modelo intermediário para extrair features
    if layer_names is None:
        # Pegar as primeiras camadas convolucionais
        layer_names = []
        for layer in model.layers:
            if 'conv' in layer.name:
                layer_names.append(layer.name)
                if len(layer_names) >= 4:  # Limitar a 4 camadas
                    break
    
    # Criar modelo que retorna outputs das camadas selecionadas
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    
    # Obter features
    feature_maps = feature_model.predict(img)
    
    # Visualizar
    for i, feature_map in enumerate(feature_maps):
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        
        # Limitar número de features para visualização
        n_cols = min(8, n_features)
        n_rows = min(4, (n_features // n_cols) + (1 if n_features % n_cols else 0))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(f'Feature Maps - Camada: {layer_names[i]}')
        
        for j in range(min(n_features, n_rows * n_cols)):
            row, col = j // n_cols, j % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes
            elif n_cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            ax.imshow(feature_map[0, :, :, j], cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Ocultar eixos extras
        for j in range(min(n_features, n_rows * n_cols), n_rows * n_cols):
            row, col = j // n_cols, j % n_cols
            if n_rows == 1:
                axes[col].axis('off') if n_cols > 1 else axes.axis('off')
            elif n_cols == 1:
                axes[row].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

def visualize_filters(model, layer_name):
    """Visualiza filtros de uma camada convolucional"""
    layer = model.get_layer(layer_name)
    
    if 'conv' not in layer.name.lower():
        print(f"A camada {layer_name} não é uma camada convolucional")
        return
    
    # Obter pesos dos filtros
    weights = layer.get_weights()[0]  # [filtros, biases]
    
    print(f"Formato dos filtros: {weights.shape}")
    
    # Visualizar filtros (assumindo formato [filter_height, filter_width, input_channels, output_channels])
    n_filters = min(64, weights.shape[-1])  # Limitar visualização
    n_cols = 8
    n_rows = (n_filters // n_cols) + (1 if n_filters % n_cols else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2*n_rows))
    fig.suptitle(f'Filtros da camada: {layer_name}')
    
    for i in range(n_filters):
        row, col = i // n_cols, i % n_cols
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Pegar filtro (média se tiver múltiplos canais de entrada)
        if weights.shape[2] == 1:  # Escala de cinza
            filter_img = weights[:, :, 0, i]
        elif weights.shape[2] == 3:  # RGB
            filter_img = np.mean(weights[:, :, :, i], axis=2)
        else:  # Outros casos
            filter_img = np.mean(weights[:, :, :, i], axis=2)
        
        ax.imshow(filter_img, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ocultar eixos extras
    for i in range(n_filters, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if n_rows == 1:
            axes[col].axis('off') if n_cols > 1 else axes.axis('off')
        elif n_cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Exemplo de Uso

```python
# src/examples/cnn_practical_example.py
from models.cnn_transfer_learning import CNNTransferLearning
from utils.cnn_visualization import visualize_feature_maps, visualize_filters
import tensorflow as tf
import numpy as np

def demonstrate_cnn_transfer_learning():
    """Demonstra transfer learning com CNN"""
    print("=== Demonstração de Transfer Learning com CNN ===\n")
    
    # Criar modelo com transfer learning
    cnn_tl = CNNTransferLearning(
        base_model_name='mobilenetv2',
        input_shape=(224, 224, 3),
        num_classes=3  # Exemplo: 3 classes
    )
    
    print("Modelo base carregado:", cnn_tl.base_model_name)
    print("Formato de entrada:", cnn_tl.input_shape)
    print("Número de classes:", cnn_tl.num_classes)
    
    # Construir modelo
    model = cnn_tl.build_model()
    print("\nModelo construído com sucesso!")
    
    # Compilar modelo
    cnn_tl.compile_model(learning_rate=0.0001)
    print("Modelo compilado")
    
    # Visualizar arquitetura
    cnn_tl.visualize_model_architecture()
    
    # Simular datasets (em uma aplicação real, você usaria seus dados reais)
    print("\nSimulando datasets para demonstração...")
    
    # Criar dados simulados para demonstração
    train_images = tf.random.normal((100, 224, 224, 3))
    train_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, size=(100,)), num_classes=3)
    val_images = tf.random.normal((20, 224, 224, 3))
    val_labels = tf.keras.utils.to_categorical(np.random.randint(0, 3, size=(20,)), num_classes=3)
    
    # Criar datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(16)
    validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(16)
    
    print("Datasets criados")
    
    # Treinar modelo inicialmente
    print("\nTreinando cabeça personalizada...")
    history = cnn_tl.train_initial(train_dataset, validation_dataset, epochs=2)  # Poucas épocas para demonstração
    
    print("Treinamento inicial concluído!")
    
    # Visualizar histórico de treinamento
    cnn_tl.plot_training_history()
    
    # Preparar para fine-tuning
    print("\nPreparando para fine-tuning...")
    cnn_tl.unfreeze_model(fine_tune_from_layer=50)
    
    # Compilar novamente com learning rate menor
    cnn_tl.compile_model(learning_rate=0.0001/10)
    
    # Fine-tuning
    print("Realizando fine-tuning...")
    fine_tune_history = cnn_tl.fine_tune(train_dataset, validation_dataset, epochs=2)
    
    print("Fine-tuning concluído!")
    
    return cnn_tl

def visualize_cnn_features():
    """Visualiza features extraídas pela CNN"""
    # Criar modelo simples para visualização
    cnn_tl = CNNTransferLearning(
        base_model_name='mobilenetv2',
        input_shape=(224, 224, 3),
        num_classes=3
    )
    model = cnn_tl.build_model()
    
    # Criar imagem de exemplo
    sample_image = tf.random.normal((224, 224, 3))
    
    print("Visualizando feature maps...")
    # Para esta demonstração, vamos visualizar algumas camadas específicas
    layer_names = []
    for layer in model.layers:
        if 'conv' in layer.name and len(layer_names) < 2:
            layer_names.append(layer.name)
    
    if layer_names:
        print(f"Visualizando camadas: {layer_names}")
        # Nota: Esta chamada pode não funcionar com o modelo completo devido à complexidade
        # A visualização real seria feita com um modelo mais simples ou camadas específicas
    else:
        print("Nenhuma camada convolucional encontrada para visualização")

def compare_architectures():
    """Compara diferentes arquiteturas de CNN"""
    architectures = ['vgg16', 'mobilenetv2']
    results = {}
    
    print("\n=== Comparação de Arquiteturas ===")
    
    for arch in architectures:
        print(f"\nTestando {arch}...")
        
        try:
            cnn_tl = CNNTransferLearning(
                base_model_name=arch,
                input_shape=(224, 224, 3),
                num_classes=3
            )
            model = cnn_tl.build_model()
            
            params = model.count_params()
            input_shape = model.input_shape
            
            results[arch] = {
                'params': params,
                'input_shape': input_shape,
                'layers': len(model.layers)
            }
            
            print(f"  Parâmetros: {params:,}")
            print(f"  Camadas: {len(model.layers)}")
            
        except Exception as e:
            print(f"  Erro ao carregar {arch}: {e}")
    
    print(f"\nResultados da comparação:")
    for arch, data in results.items():
        print(f"  {arch}: {data['params']:,} parâmetros, {data['layers']} camadas")
    
    return results

if __name__ == "__main__":
    # Executar demonstrações
    trained_model = demonstrate_cnn_transfer_learning()
    visualize_cnn_features()
    comparison_results = compare_architectures()
    
    print("\n=== Resumo da Aula ===")
    print("Hoje aprendemos:")
    print("- Como funcionam as CNNs de forma intuitiva")
    print("- Transfer learning com modelos pré-treinados")
    print("- Fine-tuning para adaptação a novos domínios")
    print("- Visualização de features e arquiteturas")
    print("- Comparação entre diferentes arquiteturas")