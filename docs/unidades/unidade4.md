# Unidade 4 - Deep Learning Aplicado

## Introdução às Redes Neurais Convolucionais (CNNs)

As Redes Neurais Convolucionais (Convolutional Neural Networks - CNNs) são uma classe especializada de redes neurais projetadas especificamente para processar dados com uma topologia de grade, como imagens. Elas têm sido extremamente bem-sucedidas em tarefas de visão computacional.

### Estrutura Básica de uma CNN

Uma CNN típica é composta por:

1. **Camadas Convolucionais**: Extraem características da imagem
2. **Camadas de Pooling**: Reduzem a dimensionalidade espacial
3. **Camadas totalmente conectadas (Dense)**: Realizam a classificação final
4. **Funções de ativação**: Introduzem não-linearidade

### Como Funciona uma Camada Convolucional

A convolução é uma operação matemática que aplica um filtro (ou kernel) sobre uma imagem para detectar características específicas:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Exemplo de camada convolucional
def visualize_conv_layer():
    # Criar uma imagem de exemplo (1 imagem, 28x28 pixels, 1 canal)
    sample_image = np.random.rand(1, 28, 28, 1).astype(np.float32)
    
    # Criar camada convolucional
    conv_layer = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
    
    # Aplicar convolução
    output = conv_layer(sample_image)
    
    print(f"Entrada: {sample_image.shape}")
    print(f"Saída: {output.shape}")
    
    return conv_layer

conv_layer = visualize_conv_layer()
```

### Arquitetura Típica de uma CNN

```python
def create_basic_cnn(input_shape, num_classes):
    """Cria uma CNN básica para classificação"""
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Camadas densas para classificação
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Regularização
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Exemplo de uso
model = create_basic_cnn((32, 32, 3), 10)  # CIFAR-10
model.summary()
```

## Transfer Learning

O Transfer Learning é uma técnica poderosa que permite reutilizar um modelo pré-treinado em uma nova tarefa, economizando tempo e recursos computacionais.

### Por que usar Transfer Learning?

- **Eficiência**: Treinar uma CNN do zero requer grandes conjuntos de dados e tempo computacional
- **Performance**: Modelos pré-treinados já aprenderam características genéricas de imagens
- **Adaptabilidade**: Ajustar um modelo existente para novas classes específicas

### Implementação de Transfer Learning

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_transfer_model(num_classes, input_shape=(224, 224, 3)):
    """Cria modelo usando transfer learning com VGG16"""
    # Carregar modelo pré-treinado sem a cabeça de classificação
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Congelar pesos do modelo base
    base_model.trainable = False
    
    # Adicionar camadas personalizadas
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

# Exemplo de uso
transfer_model, base_model = create_transfer_model(num_classes=5)
transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

transfer_model.summary()
```

### Fine-tuning

Depois de treinar a cabeça de classificação personalizada, podemos desfrizar parte do modelo base e continuar o treinamento:

```python
def fine_tune_model(model, base_model, train_dataset, validation_dataset, epochs=10):
    """Realiza fine-tuning do modelo"""
    # Desfrizar parte do modelo base
    base_model.trainable = True
    
    # Congelar as primeiras camadas (menos específicas)
    fine_tune_at = len(base_model.layers) // 2
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Compilar com learning rate menor
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar novamente
    history_fine = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )
    
    return history_fine

# Exemplo de uso (após treinamento inicial)
# history_fine = fine_tune_model(transfer_model, base_model, train_ds, val_ds, epochs=5)
```

## Detecção de Objetos com YOLO

YOLO (You Only Look Once) é uma arquitetura popular para detecção de objetos em tempo real.

### Conceitos de Detecção de Objetos

Ao contrário da classificação de imagens, a detecção de objetos envolve:

- **Localização**: Identificar onde os objetos estão (bounding boxes)
- **Classificação**: Determinar que tipo de objeto é
- **Confiança**: Estimar a certeza da detecção

### Estrutura de Dados para Detecção

```python
import numpy as np

def create_detection_example():
    """Exemplo de estrutura de dados para detecção de objetos"""
    # Suponha uma imagem 416x416 com 2 objetos detectados
    detections = {
        'boxes': np.array([  # [x_min, y_min, x_max, y_max]
            [50, 60, 150, 200],   # Objeto 1
            [200, 100, 300, 250]  # Objeto 2
        ]),
        'labels': np.array(['pessoa', 'carro']),
        'scores': np.array([0.95, 0.87])  # Confiança da detecção
    }
    
    return detections

detections = create_detection_example()
print("Exemplo de detecções:")
for i in range(len(detections['boxes'])):
    box = detections['boxes'][i]
    label = detections['labels'][i]
    score = detections['scores'][i]
    print(f"  {label}: {box} (confiança: {score:.2f})")
```

### Implementação Básica de Detecção com TensorFlow/Keras

```python
def create_object_detection_model(num_classes, input_shape=(416, 416, 3)):
    """Cria modelo básico para detecção de objetos"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Backbone CNN (feature extractor)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, 
        include_top=False, 
        weights='imagenet'
    )(inputs)
    
    # Adicionar camadas para detecção
    x = layers.GlobalAveragePooling2D()(backbone)
    x = layers.Dense(1024, activation='relu')(x)
    
    # Saída para bounding boxes (4 coordenadas)
    bbox_output = layers.Dense(4, name='bbox')(x)
    
    # Saída para classificação
    class_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    model = Model(inputs=inputs, outputs=[bbox_output, class_output])
    
    return model

# Exemplo de uso
detection_model = create_object_detection_model(num_classes=10)
detection_model.compile(
    optimizer='adam',
    loss={
        'bbox': 'mse',
        'classification': 'categorical_crossentropy'
    },
    metrics={
        'classification': 'accuracy'
    }
)

detection_model.summary()
```

### Uso de Modelos Pré-Treinados para Detecção

```python
import cv2
import numpy as np

def load_pretrained_yolo():
    """Carrega modelo YOLO pré-treinado (exemplo conceitual)"""
    # Na prática, usaria uma biblioteca como ultralytics/yolov5 ou tensorflow-models
    print("Carregando modelo YOLO pré-treinado...")
    
    # Exemplo com OpenCV DNN (modelo YOLO carregado como Darknet)
    # net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    # layer_names = net.getLayerNames()
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Para este exemplo, retornaremos um placeholder
    class YOLODetector:
        def detect(self, image):
            # Simular detecção
            height, width = image.shape[:2]
            
            # Gerar detecções simuladas
            boxes = [
                [int(0.1 * width), int(0.1 * height), int(0.3 * width), int(0.3 * height)],
                [int(0.6 * width), int(0.4 * height), int(0.8 * width), int(0.7 * height)]
            ]
            confidences = [0.9, 0.75]
            class_ids = [0, 1]  # Ex: pessoa, carro
            
            return boxes, confidences, class_ids
    
    return YOLODetector()

def draw_detections(image, boxes, confidences, class_ids, classes):
    """Desenha detecções na imagem"""
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

# Exemplo de uso
detector = load_pretrained_yolo()
classes = ['pessoa', 'carro', 'bicicleta', 'cachorro', 'gato']

# Simular imagem
sample_img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

# Detectar objetos
boxes, confidences, class_ids = detector.detect(sample_img)

# Desenhar detecções
result_img = draw_detections(sample_img.copy(), boxes, confidences, class_ids, classes)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_img)
plt.title('Imagem Original')
plt.subplot(1, 2, 2)
plt.imshow(result_img)
plt.title('Detecções')
plt.show()
```

## Otimização e Deploy

### Quantização de Modelos

A quantização reduz o tamanho e melhora a velocidade de execução de modelos, especialmente importante para deploy em dispositivos com recursos limitados:

```python
def quantize_model(saved_model_dir):
    """Converte modelo para versão quantizada"""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Quantização inteira
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Converter para TensorFlow Lite
    quantized_model = converter.convert()
    
    return quantized_model

def save_quantized_model(quantized_model, filename):
    """Salva modelo quantizado"""
    with open(filename, 'wb') as f:
        f.write(quantized_model)

# Exemplo de uso (após treinamento do modelo)
# quantized_tflite_model = quantize_model('saved_model_dir')
# save_quantized_model(quantized_tflite_model, 'model_quantized.tflite')
```

### Exportação para TensorFlow Lite

TensorFlow Lite é otimizado para inferência em dispositivos móveis e embarcados:

```python
def export_to_tflite(model, filename):
    """Exporta modelo para TensorFlow Lite"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Opcional: habilitar otimizações
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Salvar modelo
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modelo TFLite salvo como {filename}")

# Exemplo de uso
# export_to_tflite(transfer_model, 'my_model.tflite')
```

### Inferência com TensorFlow Lite

```python
def run_tflite_inference(interpreter, input_data):
    """Executa inferência com modelo TFLite"""
    # Obter entrada e saída do modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preparar entrada
    input_shape = input_details[0]['shape']
    input_data = input_data.reshape(input_shape).astype(np.float32)
    
    # Definir entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Executar inferência
    interpreter.invoke()
    
    # Obter saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

# Exemplo de uso (após carregar modelo TFLite)
# interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
# interpreter.allocate_tensors()
# prediction = run_tflite_inference(interpreter, sample_input)
```

## Avaliação de Modelos de Deep Learning

### Métricas para Classificação

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_classification_model(model, test_dataset):
    """Avalia modelo de classificação"""
    # Obter previsões
    predictions = model.predict(test_dataset)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Obter rótulos verdadeiros
    true_classes = []
    for _, labels in test_dataset:
        true_classes.extend(labels.numpy())
    true_classes = np.array(true_classes)
    
    # Calcular métricas
    report = classification_report(true_classes, predicted_classes)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    print("Relatório de Classificação:")
    print(report)
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()
    
    return report, cm

# Exemplo de uso
# report, cm = evaluate_classification_model(transfer_model, test_dataset)
```

### Métricas para Detecção de Objetos

```python
def calculate_iou(box1, box2):
    """Calcula Intersection over Union entre duas bounding boxes"""
    # Coordenadas: [x_min, y_min, x_max, y_max]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calcular interseção
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Área de interseção
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height
    
    # Áreas das bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # União
    union_area = box1_area + box2_area - intersection_area
    
    # IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou

def evaluate_object_detection(predictions, ground_truth, iou_threshold=0.5):
    """Avalia modelo de detecção de objetos"""
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    for pred_box, pred_label in predictions:
        matched = False
        for gt_box, gt_label in ground_truth:
            if pred_label == gt_label:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp += 1
                    matched = True
                    break
        
        if not matched:
            fp += 1
    
    # Calcular FN
    fn = len(ground_truth) - tp
    
    # Calcular métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    
    return precision, recall, f1_score

# Exemplo de uso
# predictions = [([50, 60, 150, 200], 'pessoa'), ([200, 100, 300, 250], 'carro')]
# ground_truth = [([55, 65, 145, 195], 'pessoa'), ([195, 95, 305, 255], 'carro')]
# precision, recall, f1 = evaluate_object_detection(predictions, ground_truth)
```

## Considerações Práticas

### Escolha de Arquitetura

- **VGG**: Simples mas pesado, bom para aprendizado
- **ResNet**: Excelente desempenho, permite redes muito profundas
- **MobileNet**: Otimizado para dispositivos móveis
- **EfficientNet**: Bom equilíbrio entre acurácia e eficiência

### Data Augmentation

Técnica importante para aumentar o conjunto de treinamento e melhorar a generalização:

```python
def create_data_augmentation():
    """Cria camadas de augmentation"""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1)
    ])
    
    return data_augmentation

# Exemplo de uso
augmentation_layer = create_data_augmentation()

# Aplicar durante treinamento
# model = tf.keras.Sequential([
#     augmentation_layer,
#     base_model,
#     # ... outras camadas
# ])
```

### Estratégias de Treinamento

- **Learning Rate Scheduling**: Ajustar taxa de aprendizado durante o treinamento
- **Early Stopping**: Parar treinamento quando o desempenho estagnar
- **Checkpointing**: Salvar melhores versões do modelo durante treinamento

```python
def setup_training_callbacks():
    """Configura callbacks para treinamento"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

# Exemplo de uso
# callbacks = setup_training_callbacks()
# model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
```

O Deep Learning revolucionou a visão computacional, permitindo soluções mais precisas e robustas para uma ampla variedade de tarefas. O uso de técnicas como transfer learning torna possível aplicar essas tecnologias mesmo com conjuntos de dados menores.