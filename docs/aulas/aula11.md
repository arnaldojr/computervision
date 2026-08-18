## Vision Transformers (ViT)

Material da Aula 11 — primeiro vamos construir a ideia de um Vision Transformer; depois conectaremos essa ideia ao PyTorch e ao transfer learning.

Ao final desta primeira parte, você deverá conseguir explicar como uma imagem vira uma sequência de tokens e calcular quantos tokens são produzidos. Bibliotecas como `timm` e Hugging Face aparecem somente depois dessa etapa.

[Lab 13 — Vision Transformers (download)](lab13/lab13.zip){ .md-button .md-button--primary }
    
[Abrir no Google Colab](https://colab.research.google.com/github/arnaldojr/computervision/blob/main/docs/aulas/lab13/notebook_vit2.ipynb){ .md-button .md-button--primary }

[Baixar notebook](lab13/notebook_vit2.ipynb){ .md-button download="notebook_vit2.ipynb" }


---

### Uma imagem pode virar uma sequência?

Antes de falar em modelos, observe a transformação abaixo. O simulador usa uma imagem pequena para deixar explícito o que normalmente fica escondido dentro de uma implementação.

<div id="vit-patches-widget" class="vit-widget" markdown="1">
<div class="vit-widget__intro">
    <strong>Experimente a ideia central do ViT</strong>
    <span>Escolha um exemplo ou desenhe na imagem. Depois clique em um patch para acompanhar sua posição na sequência.</span>
</div>
<div class="vit-widget__controls">
    <button type="button" data-vit-example="object">Exemplo: objeto</button>
    <button type="button" data-vit-example="diagonal">Exemplo: diagonal</button>
    <button type="button" data-vit-clear>Limpar</button>
    <label for="vit-patch-size">Tamanho do patch</label>
    <input id="vit-patch-size" data-vit-patch-size type="range" min="2" max="8" step="2" value="4">
    <output data-vit-patch-value>4 × 4 pixels</output>
</div>
<div class="vit-widget__boards">
    <figure>
        <canvas data-vit-input width="320" height="320" aria-label="Imagem para desenhar"></canvas>
        <figcaption>1. A imagem de entrada</figcaption>
    </figure>
    <figure>
        <canvas data-vit-patches width="320" height="320" aria-label="Imagem dividida em patches"></canvas>
        <figcaption>2. A imagem dividida em patches</figcaption>
    </figure>
    <figure class="vit-widget__tokens">
        <canvas data-vit-tokens width="640" height="120" aria-label="Sequência de tokens"></canvas>
        <figcaption>3. Os patches são organizados como uma sequência</figcaption>
    </figure>
</div>
<div class="vit-widget__readout">
    <span data-vit-status></span>
    <strong data-vit-selected></strong>
    <span data-vit-coordinates></span>
    <span><b data-vit-patch-count></b> | imagem: <b data-vit-image-shape></b> | sequência: <b data-vit-token-shape></b></span>
</div>
</div>

O ponto importante não é a imagem desenhada. É a mudança de representação:

1. começamos com pixels organizados em uma grade;
2. dividimos a grade em regiões menores, os **patches**;
3. colocamos essas regiões em uma ordem;
4. cada região será transformada em um vetor, chamado **token**.

!!! question "Pare e preveja"
        Se uma imagem tiver tamanho $224 \times 224$ e cada patch medir $16 \times 16$, quantos patches aparecerão na sequência? Use o simulador e depois confira:

        $$
        N = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196
        $$

O simulador ainda não está fazendo classificação nem atenção. Essa separação é proposital: primeiro precisamos entender a entrada do Transformer. A próxima pergunta será como o modelo sabe de onde veio cada token.

### De CNNs para Transformers

CNNs dominaram visão computacional por uma década. Em 2020, o artigo **"An Image is Worth 16×16 Words"** mostrou que Transformers puros, sem convoluções, alcançam resultados equivalentes ou superiores — especialmente em grandes datasets.

```
CNN                              Vision Transformer (ViT)
────────────────────             ──────────────────────────────
Varredura local (kernel)   →     Atenção global (todos os patches)
Hierarquia de features     →     Features globais desde a 1ª camada
Induction bias espacial    →     Sem induction bias → precisa de mais dados
```

!!! info "Por que ViTs importam?"
    Modelos modernos de visão (CLIP, SAM, DINO, LLaVA) são baseados em Transformers.
    Entender ViTs é o primeiro passo para trabalhar com modelos multimodais.

---

### Como funciona o ViT

=== "1 — Patch Embedding"

    A imagem é dividida em patches fixos (ex: 16×16 pixels) e cada patch é linearizado e projetado num vetor de embedding — como palavras em NLP.

    $$
    \text{imagem } (H \times W \times C) \rightarrow N \text{ patches} \rightarrow N \text{ tokens}
    $$

    Para uma imagem 224×224 com patches 16×16:

    $$
    N = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196 \text{ tokens}
    $$

=== "2 — Positional Encoding"

    Como Transformers não têm senso de posição, adicionamos **embeddings posicionais** learnable a cada token, preservando a informação espacial.

=== "3 — Multi-Head Self-Attention"

    Cada token "presta atenção" em todos os outros. A atenção mede a relevância entre pares de tokens:

    $$
    \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

    Com múltiplas "cabeças" (heads), o modelo captura diferentes tipos de relação entre patches.

=== "4 — CLS Token + Classificador"

    Um token especial `[CLS]` é concatenado aos patches. Após passar pelos blocos Transformer, o estado final do `[CLS]` é usado como representação global da imagem para classificação.

---

### Família de modelos

| Modelo | Patches | Parâmetros | ImageNet Top-1 | Indicado para |
|--------|---------|------------|----------------|---------------|
| DeiT-Tiny | 16×16 | 5.7M | 72.2% | rápido, CPU-friendly |
| DeiT-Small | 16×16 | 22M | 79.8% | equilíbrio |
| DeiT-Base | 16×16 | 86M | 81.8% | alta acurácia |
| ViT-B/16 | 16×16 | 86M | 81.1% | original |
| Swin-Tiny | janelas | 28M | 81.3% | eficiente para alta resolução |

!!! tip "DeiT vs ViT"
    **DeiT** (Data-efficient Image Transformers) foi treinado apenas com ImageNet (~1M imagens), sem dados extras.
    Usa **knowledge distillation** de uma CNN para compensar a falta de induction bias.

---

### ViT vs CNN — comparativo prático

=== "Acurácia"

    | Modelo | Parâmetros | CIFAR-10 (fine-tuning) |
    |--------|-----------|----------------------|
    | CNN do zero (Lab 10) | 2.5M | ~70–75% |
    | MobileNetV3 TL (Lab 12) | 2.5M | ~85–90% |
    | DeiT-Tiny TL (Lab 13) | 5.7M | ~92–94% |

=== "Quando usar ViT"

    - Dataset grande (>10k imagens por classe) — ViTs brilham com escala
    - Quando precisar de features globais (relações entre regiões distantes)
    - Pipelines multimodais (texto + imagem)

=== "Quando usar CNN"

    - Dataset pequeno — CNNs têm mais induction bias espacial
    - Edge devices com restrição de memória/latência
    - Quando a latência é crítica (CNNs compactas ainda ganham em velocidade)

---

### Fine-tuning com HuggingFace Transformers

=== "Carregar modelo"

    ```python
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")
    model = AutoModelForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True,  # substitui o head original (1000 classes)
    )
    ```

=== "Pré-processamento"

    O `AutoImageProcessor` cuida do resize, normalização e conversão para tensor:

    ```python
    inputs = processor(images=pil_image, return_tensors="pt")
    # inputs["pixel_values"]: tensor [1, 3, 224, 224]
    ```

=== "Treino"

    ```python
    outputs = model(**inputs, labels=label)
    loss = outputs.loss
    loss.backward()
    ```

---

### Como executar o lab

```bash
# 1. treinar (notebook_vit.ipynb)
# Requer ~4GB de RAM — GPU recomendada mas não obrigatória (DeiT-Tiny é leve)

# 2. iniciar a API
pip install -r requirements.txt
uvicorn app:app --reload

# 3. acessar
# http://localhost:8000        → interface web
# http://localhost:8000/docs   → Swagger UI
```

---

### Referências

- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877)
- [HuggingFace — Image Classification](https://huggingface.co/docs/transformers/tasks/image_classification)
- [timm — PyTorch Image Models](https://huggingface.co/docs/timm)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
