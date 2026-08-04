## Transfer Learning com CNNs

Material da Aula 10 — Transfer Learning com redes convolucionais pré-treinadas.
Reutilizamos os pesos de um modelo treinado no ImageNet para classificar um novo dataset com muito menos dados e tempo.

[Lab 12 — Transfer Learning (download)](lab12/lab12.zip){ .md-button .md-button--primary }

---

### O que é Transfer Learning?

Treinar uma CNN do zero exige **grandes datasets** e **muito tempo**. Transfer Learning contorna isso: pegamos um modelo já treinado em um dataset enorme (ImageNet, 1.2M imagens) e **reutilizamos o conhecimento** aprendido.

```
ImageNet (1.2M imagens)          Seu dataset (centenas de imagens)
     ↓                                     ↓
 Backbone pré-treinado   →   Fine-tuning das últimas camadas
 (features genéricas)         (features específicas do domínio)
```

!!! tip "Intuição"
    As primeiras camadas de uma CNN aprendem **bordas, texturas e formas básicas** — úteis para qualquer domínio visual.
    Só as últimas camadas (o classificador) precisam ser re-treinadas para o novo problema.

---

### Estratégias de Transfer Learning

=== "Feature Extraction"

    Congela **todas** as camadas do backbone. Treina apenas o classificador (head).

    - Mais rápido
    - Melhor quando seu dataset é pequeno e similar ao ImageNet
    - Menor risco de overfitting

    ```python
    for param in model.features.parameters():
        param.requires_grad = False  # congela backbone
    # apenas model.classifier treina
    ```

=== "Fine-Tuning"

    Descongela **parte** do backbone + treina o classificador.

    - Mais lento, mas geralmente mais preciso
    - Melhor quando seu dataset é maior ou muito diferente do ImageNet
    - Usar learning rate menor no backbone para não destruir os pesos

    ```python
    # backbone com lr baixo, head com lr normal
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ])
    ```

=== "Comparativo"

    | | Feature Extraction | Fine-Tuning |
    |--|--|--|
    | Velocidade | ⚡ Rápido | 🐢 Mais lento |
    | Dados necessários | Poucos | Mais |
    | Risco de overfitting | Baixo | Médio |
    | Acurácia típica | Boa | Melhor |
    | Quando usar | Dataset pequeno, similar ao ImageNet | Dataset maior ou diferente |

---

### MobileNetV3 para Transfer Learning

=== "Arquitetura"

    ```python
    import torchvision.models as models

    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Estrutura:
    # model.features  → backbone (extrator de features)
    # model.avgpool   → pooling global
    # model.classifier → [Linear, Hardswish, Dropout, Linear(1000)]
    ```

=== "Substituindo o classifier"

    ```python
    num_classes = 10  # seu número de classes

    # Substitui apenas a última camada linear
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    ```

=== "Comparativo de modelos"

    | Modelo | Parâmetros | Top-1 ImageNet | Indicado para |
    |--------|-----------|----------------|---------------|
    | MobileNetV3-Small | 2.5M | 67.7% | dispositivos móveis, inferência rápida |
    | MobileNetV3-Large | 5.5M | 75.3% | equilíbrio velocidade/acurácia |
    | ResNet-18 | 11.7M | 69.8% | baseline clássico |
    | ResNet-50 | 25.6M | 76.1% | alta acurácia, mais VRAM |
    | EfficientNet-B0 | 5.3M | 77.7% | melhor acurácia/parâmetro |

---

### Comparação: Do zero vs Transfer Learning

No lab12 você vai repetir a classificação do CIFAR-10 (já vista no lab10), mas agora com transfer learning:

| | Lab 10 (CNN do zero) | Lab 12 (Transfer Learning) |
|--|--|--|
| Épocas para convergir | ~20–30 | ~5–10 |
| Acurácia final | ~70–75% | ~85–90% |
| Parâmetros treináveis | Todos (2.5M) | Só o head (~41k) |
| Tempo por época | Similar | Similar ou menor |

!!! warning "CIFAR-10 e Transfer Learning"
    As imagens do CIFAR-10 são 32×32 pixels — bem menores do que os 224×224 que o ImageNet usa.
    No lab vamos redimensionar via `transforms.Resize(224)` antes de passar ao modelo.

---

### Como executar o lab

```bash
# 1. treinar o modelo (notebook)
# Execute notebook_transfer_learning.ipynb até a célula "Salvar artefato"

# 2. copiar artifacts/ para a pasta da API
# os arquivos mobilenet_cifar10.pt e metadata.json são gerados no notebook

# 3. iniciar a API
pip install -r requirements.txt
uvicorn app:app --reload
```

---

### Referências

- [torchvision — Transfer Learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [MobileNetV3 paper](https://arxiv.org/abs/1905.02244)
- [ResNet paper — Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [CS231n — Transfer Learning](https://cs231n.github.io/transfer-learning/)
