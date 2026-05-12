## Introdução às CNNs


Material da Aula 8 de Deep Learning + API de Visão Computacional.
Treinamento de uma CNN no PyTorch com CIFAR-10 e deploy do modelo via FastAPI, com interface web para upload de imagem.

Vamos manter o mesmo estilo de projeto que voce já conhece no curso:

[Notebook-CNN + API FastAPI (download)](lab10/api/02_api_fastapi.zip){ .md-button .md-button-primary }


### o que são CNNs e para que servem?


As **Redes Neurais Convolucionais (CNNs)** são um tipo de rede neural artificial, projetada para processar dados que possuem uma **estrutura topológica similar a uma grade**, como:

Aplicações comuns:

- Classificação e segmentação de imagens
- Reconhecimento facial e detecção de objetos
- Análise de sinais e séries temporais
- Bioinformática (motivos em DNA/RNA)

### Vantagens sobre MLPs

| Aspecto | MLP | CNN |
|---------|-----|-----|
| Parâmetros | Crescem explosivamente | Muito menos (filtros reutilizados) |
| Estrutura espacial | Perdida | Preservada |
| Robustez a deslocamentos | Baixa | Maior (quase invariante a translação) |
| Compartilhamento de pesos | Não | Sim |
| Escalabilidade em visão | Limitada | Alta |

![alt text](lab10/imgs/same_padding_no_strides.gif)


---

### Convolução (Intuição)

A convolução mede o alinhamento entre um pequeno padrão (kernel) e regiões da entrada. 

![alt text](lab10/imgs/convnet.png)

=== "Intuição"

    O kernel "varre" a imagem e produz um mapa de ativações (feature map).
    Regiões que se parecem com o padrão do kernel geram respostas mais fortes.

=== "Contínua"

    $$
    (f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)\,d\tau
    $$

=== "Discreta (CNNs)"

    $$
    (f * g)[n] = \sum_{m=-\infty}^{\infty} f[m]g[n-m]
    $$

=== "2D para Imagens"

    Em visão computacional, usa-se tecnicamente correlação cruzada (sem inversão do kernel), mas o nome convolução é mantido por convenção.

    $$
    S(i,j) = (I * K)(i,j) = \sum_{m}\sum_{n} I(i+m, j+n)\,K(m,n)
    $$

    Onde:

    - `I`: imagem de entrada
    - `K`: kernel (filtro)
    - `S`: feature map
    - `m, n`: índices do kernel
    - `i, j`: índices da posição atual na imagem de entrada


    ![Convolução 3D](lab10/imgs/conv3d.gif)

### Exemplo Prático de Convolução

=== "Entrada e Kernel"

    **Imagem 5x5:**

    $$
    I = \begin{bmatrix}
    1 & 2 & 3 & 0 & 1 \\\\
    0 & 1 & 2 & 3 & 1 \\\\
    1 & 0 & 1 & 2 & 0 \\\\
    2 & 1 & 0 & 1 & 2 \\\\
    1 & 0 & 2 & 1 & 0
    \end{bmatrix}
    $$

    **Kernel 3x3 (detector de borda):**

    $$
    K = \begin{bmatrix}
    -1 & -1 & -1 \\\\
    -1 & 8 & -1 \\\\
    -1 & -1 & -1
    \end{bmatrix}
    $$

=== "Cálculo de S(1,1)"

    $$
    \begin{aligned}
    S(1,1) &= (-1\cdot1) + (-1\cdot2) + (-1\cdot3) \\
            &\quad + (-1\cdot0) + (8\cdot1) + (-1\cdot2) \\
            &\quad + (-1\cdot1) + (-1\cdot0) + (-1\cdot1) \\
            &= -5
    \end{aligned}
    $$

---

## Parametros da Camada Convolucional

Uma camada convolucional não apenas aplica filtros sobre a imagem. Seus parâmetros definem **o que será observado**, **como o filtro se desloca** e **qual será o tamanho da saída**.

=== "Kernel/Filtro"

    O **kernel** é a pequena matriz de pesos que percorre a imagem procurando padrões locais, como bordas, texturas e formas simples.

=== "Stride"

    O **stride** define o tamanho do passo do kernel ao percorrer a imagem.

=== "Padding"

    O **padding** adiciona bordas artificiais à imagem antes da convolução. Ele é usado para controlar a perda de dimensão nas bordas.

<quiz>
Efeito de padding='valid' com kernel 3×3 e stride=1 em H×W?
- [ ] Aumenta tamanho
- [x] Reduz 2 pixels (1 por borda)
- [ ] Não altera
- [ ] Dobra dimensões

Sem padding, a janela não cobre bordas externas totalmente, reduzindo largura e altura em 1 de cada lado.
</quiz>

<quiz>
Principal efeito de stride=2 em convolução?
- [ ] Aumentar resolução espacial
- [x] Diminuir resolução e custo computacional
- [ ] Substituir função de ativação
- [ ] Tornar o kernel maior

Stride>1 “pula” posições, gerando feature maps menores e operação mais barata.
</quiz>


---

## Tipos de Convoluções

### Convolução Standard

Além da convolução padrão, existem variações que modificam **como os filtros operam sobre os canais**, **como ampliam o campo de visão** ou **como aumentam a resolução espacial**.

=== "Standard"

    A convolução padrão aplica vários filtros sobre todos os canais da entrada.  
    Em uma imagem RGB, cada filtro observa os 3 canais ao mesmo tempo.

    ```python title="conv_standard.py" linenums="1" hl_lines="2 3 4"
    nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1
    )
    ```

    Nesse exemplo, a camada recebe uma imagem com 3 canais e gera 32 mapas de características.

    ```text
    Entrada: [N, 3, H, W]
    Saída:   [N, 32, H, W]
    ```

    O `padding=1` preserva altura e largura quando usamos `kernel_size=3` e `stride=1`.

=== "Depthwise Separable"

    A convolução separável em profundidade divide a operação em duas etapas:

    1. **Depthwise:** aplica filtros espaciais em cada canal separadamente.
    2. **Pointwise:** usa convoluções `1x1` para combinar os canais e gerar novos mapas de características.

    ```python title="conv_depthwise_pointwise.py" linenums="1" hl_lines="6 9 15"
    import torch.nn as nn

    depthwise = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=3
    )

    pointwise = nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=1
    )
    ```

    Na etapa **depthwise**, o parâmetro `groups=3` faz com que cada canal da imagem RGB seja processado separadamente.

    ```text
    Entrada:   [N, 3, H, W]
    Depthwise: [N, 3, H, W]
    Pointwise: [N, 32, H, W]
    ```

    A etapa **pointwise** usa `kernel_size=1` para misturar os canais em cada posição espacial e produzir os 32 mapas de características finais.


=== "Dilatada (Atrous)"

    A convolução dilatada aumenta o campo de visão do filtro sem aumentar o número de pesos.

    ```python title="conv_dilatada.py" linenums="1" hl_lines="1 7"
    nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2
    )
    ```

    O parâmetro essencial é `dilation=2`. Ele cria espaçamentos entre os elementos do kernel.

    ```text
    Kernel 3x3 normal:     observa vizinhos próximos
    Kernel 3x3 dilatado:   observa uma região maior
    ```

    É útil quando queremos capturar contexto mais amplo sem reduzir a resolução espacial, como em segmentação semântica.

=== "Transposta"

    A convolução transposta é usada para aumentar a dimensão espacial de um feature map.

    ```python title="conv_transposta.py" linenums="1" hl_lines="1 5"
    nn.ConvTranspose2d(
        in_channels=32,
        out_channels=3,
        kernel_size=3,
        stride=2
    )
    ```

    O `ConvTranspose2d` faz uma operação de upsampling aprendível.  
    Com `stride=2`, a tendência é aumentar a altura e a largura da saída.

    ```text
    Entrada: feature map menor
    Saída:   feature map maior
    ```

    É comum em autoencoders, GANs e redes de segmentação.

---

### Visualização da Convolução

<div id="cnn-widget" style="max-width:1100px;margin:1rem 0;padding:1rem;border:1px solid #e5e7eb;border-radius:12px;background:var(--md-default-bg-color,#fff)">
  <h3 style="margin-top:0">CNN – Convolução, Ativação e Pooling (interativo)</h3>

  <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-start">
    <!-- Coluna esquerda: entrada/desenho -->
    <div style="flex:1 1 260px">
      <div style="display:flex;gap:.5rem;align-items:center;margin-bottom:.5rem">
        <strong>Entrada (28×28)</strong>
        <button id="cnn_clear" class="md-button">Limpar</button>
        <button id="cnn_noise" class="md-button">Ruído</button>
        <button id="cnn_demo" class="md-button">Demo “7”</button>
      </div>
      <canvas id="cnn_input" width="196" height="196" style="image-rendering:pixelated;border:1px solid #ccc;border-radius:8px;background:#fff"></canvas>
      <div style="font-size:.85em;color:#666;margin-top:.25rem">Dica: desenhe com o mouse (clique e arraste). A imagem é 28×28, mostrada ampliada.</div>
    </div>

    <!-- Coluna centro: controles -->
    <div style="flex:1 1 280px">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem">
        <label style="grid-column:1/-1">
          <div><strong>Kernel</strong></div>
          <select id="cnn_kernel" style="width:100%">
            <option value="identity">Identity</option>
            <option value="blur">Blur (Box)</option>
            <option value="sharpen">Sharpen</option>
            <option value="edge_lap">Edge (Laplacian)</option>
            <option value="sobel_x">Sobel X</option>
            <option value="sobel_y">Sobel Y</option>
            <option value="emboss">Emboss</option>
            <option value="custom">Custom (3×3)</option>
          </select>
        </label>

        <div id="cnn_custom_wrap" style="grid-column:1/-1;display:none">
          <div style="margin:.25rem 0">Kernel 3×3 (Custom):</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.25rem">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="1">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="0">
            <input class="cnn_k" type="number" step="0.1" value="0">
          </div>
        </div>

        <label>
          <div><strong>Padding</strong></div>
          <select id="cnn_padding" style="width:100%">
            <option value="same">same</option>
            <option value="valid">valid</option>
          </select>
        </label>

        <label>
          <div><strong>Stride</strong></div>
          <input id="cnn_stride" type="number" min="1" max="4" step="1" value="1" style="width:100%">
        </label>

        <label>
          <div><strong>Ativação</strong></div>
          <select id="cnn_act" style="width:100%">
            <option value="none">None</option>
            <option value="relu">ReLU</option>
          </select>
        </label>

        <label>
          <div><strong>Pooling</strong></div>
          <select id="cnn_pool" style="width:100%">
            <option value="none">None</option>
            <option value="max">Max 2×2 (s=2)</option>
            <option value="avg">Avg 2×2 (s=2)</option>
          </select>
        </label>

        <div style="grid-column:1/-1;display:flex;gap:.5rem;margin-top:.25rem">
          <button id="cnn_apply" class="md-button md-button--primary">Aplicar</button>
          <button id="cnn_reset" class="md-button">Reset kernel</button>
        </div>

        <div style="grid-column:1/-1;font-size:.9em;color:#444">
          <div><strong>Saídas:</strong></div>
          <div>Conv: <span id="cnn_shape_conv">—</span> • Pool: <span id="cnn_shape_pool">—</span></div>
          <div>Resumo: <span id="cnn_summary">—</span></div>
        </div>
      </div>
    </div>

    <!-- Coluna direita: saídas -->
    <div style="flex:1 1 260px">
      <div style="margin-bottom:.5rem"><strong>Feature map (após conv/ativação)</strong></div>
      <canvas id="cnn_feat" width="196" height="196" style="image-rendering:pixelated;border:1px solid #ccc;border-radius:8px;background:#fff"></canvas>

      <div style="margin:.75rem 0 .5rem"><strong>Pooling (veremos a seguir)</strong></div>
      <canvas id="cnn_pool_canvas" width="196" height="196" style="image-rendering:pixelated;border:1px solid #ccc;border-radius:8px;background:#fff"></canvas>
    </div>
  </div>
</div>


---

## Pooling e Subsampling

![Pooling](lab10/imgs/poolingexp1.png)

Pooling reduz tamanho dos feature maps, melhora robustez a pequenas translações, ajuda no controle de overfitting e acelera o processamento.

=== "Max Pooling"

    ![Max pooling](lab10/imgs/pooling.png)

    ```python title="max_pooling.py" linenums="1" hl_lines="1"
    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    ```
    onde:

    - `kernel_size=(2,2)`: janela de pooling 2x2
    - `stride=(2,2)`: move a janela 2 pixels (sem sobreposição)


    Mantém a ativação mais forte da janela.

=== "Average Pooling"

    ![Average pooling](lab10/imgs/avg.png)

    ```python title="avg_pooling.py" linenums="1" hl_lines="1"
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
    ```

    onde:

    - `kernel_size=(2,2)`: janela de pooling 2x2
    - `stride=(2,2)`: move a janela 2 pixels (sem sobreposição)

    Faz média local, suavizando picos.

=== "Adaptive Average Pooling"

    ```python title="adaptive_avg_pooling.py" linenums="1" hl_lines="1"
    nn.AdaptiveAvgPool2d((1,1))
    ```

    Resume cada feature map em um único número. Substitui densas finais, reduz parâmetros.

---


<quiz>
Diferença essencial Max vs Average Pooling?
- [ ] Max reduz canais, Average aumenta canais
- [x] Max preserva picos; Average suaviza respostas
- [ ] Average não é diferenciável
- [ ] São iguais em prática

Max enfatiza presença; Average enfatiza contexto médio.
</quiz>


## Batch Normalization

=== "Batch Normalization"

    ![BatchNorm](lab10/imgs/image-2.png)

    A BatchNorm normaliza ativações por mini-batch, estabilizando o treinamento.

    ```python title="batchnorm.py" linenums="1" hl_lines="1"
    nn.BatchNorm2d(num_features=32)
    ```

    onde:
    
    - `num_features=32` indica o número de canais a serem normalizados, geralmente igual ao número de filtros da camada anterior.

    Benefícios:

    - acelera convergência
    - reduz sensibilidade à inicialização
    - permite learning rates maiores

    > Saiba mais em: [https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)



=== "Dropout"

    ![Dropout](lab10/imgs/image-3.png)

    O Dropout desativa neurônios aleatoriamente durante o treino para reduzir overfitting.

    ```python title="dropout.py" linenums="1" hl_lines="1"
    nn.Dropout(p=0.5)
    ```

    onde:

    - `p=0.5` indica a probabilidade de desativar cada neurônio durante o treinamento. Durante a inferência, o dropout é desativado e os pesos são escalados para compensar a ausência de dropout.


    > Saiba mais em: [https://www.deeplearningbook.com.br/capitulo-23-como-funciona-o-dropout/](https://www.deeplearningbook.com.br/capitulo-23-como-funciona-o-dropout/)

---

## Arquiteturas Clássicas de CNN

=== "LeNet-5 (1998)"

    [![lenet](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DFwFduRA_L6Q)](https://www.youtube.com/watch?v=FwFduRA_L6Q)

    ![alt text](lab10/imgs/lenet.png)

    ```python title="lenet5.py" linenums="1" hl_lines="5 6 7 8 14 15 16 17 21"
    import torch.nn as nn 
    class LeNet5(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = torch.tanh(self.conv1(x))
            x = self.pool1(x)
            x = torch.tanh(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, 16*5*5)  # flatten
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.softmax(self.fc3(x), dim=1)
            return x
    ```

    onde:

    - `Conv2d(1, 6, kernel_size=5)`: 1 canal de entrada (grayscale), 6 filtros, kernel 5x5
    - `AvgPool2d(kernel_size=2, stride=2)`: pooling 2x2 com stride 2
    - `Linear(16*5*5, 120)`: camada densa com 120 neurônios, recebendo a saída achatada da última camada convolucional.
    - a função de ativação usada é `tanh`, e a saída final é uma distribuição de probabilidade sobre as 10 classes usando `softmax`, o `dim=1` indica que a softmax é aplicada ao longo do último eixo (as classes).


=== "AlexNet (2012)"

    ![AlexNet](lab10/imgs/AlexNet-1.png)

    Marco histórico: consolidou uso de ReLU, Dropout e treinamento em larga escala.


    ```python title="alexnet_trecho.py" linenums="1" hl_lines="7 10 13 15 17 21 24"
    import torch.nn as nn
    class AlexNet(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            ) 
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
            return x
    ``` 

    onde: 

    - A arquitetura é composta por 5 camadas convolucionais, seguidas por 3 camadas densas. O número de filtros varia entre as camadas, começando em 96 e chegando a 384, antes de reduzir para 256 na última camada convolucional. As camadas densas têm 4096 neurônios cada, com dropout aplicado para reduzir overfitting.
    - A função de ativação usada é `ReLU`, e o método `forward` define o fluxo de dados pela rede, primeiro passando pelas camadas convolucionais e depois pelas camadas densas após achatar a saída da última camada convolucional.

=== "VGG (2014)"

    ![VGG](lab10/imgs/image-1.png)

    Estratégia: empilhar várias convoluções 3x3 + pooling.

    ```python title="vgg_trecho.py" linenums="1" hl_lines="7 9 11 13 15 17 19 21 23 25"
    import torch.nn as nn
    class VGG16(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.features = nn.Sequential(
                # Bloco 1
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Bloco 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Bloco 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Bloco 4
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Bloco 5
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 512 * 7 * 7) # flatten
            x = self.classifier(x)
            return x
    ```

    onde:

    - A arquitetura é composta por 5 blocos de convolução, cada um seguido por uma camada de pooling. O número de filtros dobra a cada bloco, começando em 64 e chegando a 512. Após as camadas convolucionais, há uma parte densa (classificador) com 3 camadas, onde as duas primeiras têm 4096 neurônios e a última camada tem `num_classes` neurônios, correspondendo ao número de classes de saída.
    - A função de ativação usada é `ReLU`, e o dropout é aplicado entre as camadas densas para reduzir o overfitting. 
    - O método `forward` define o fluxo de dados pela rede, primeiro passando pelas camadas convolucionais e depois pelas camadas densas após achatar a saída da última camada convolucional.

=== "ResNet (2015)"

    ![ResNet](lab10/imgs/image.png)

    Introduziu conexões residuais para facilitar o fluxo de gradiente em redes profundas.

    ```python title="residual_block.py" linenums="1" hl_lines="17 28 45 47 51 54 55 57"
    import torch
    import torch.nn as nn


    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()

            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.downsample = None

            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            out = self.relu(out)

            return out


    class ResNet18(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()

            self.in_channels = 64

            self.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )

            self.layer1 = self._make_layer(64,  blocks=2, stride=1)
            self.layer2 = self._make_layer(128, blocks=2, stride=2)
            self.layer3 = self._make_layer(256, blocks=2, stride=2)
            self.layer4 = self._make_layer(512, blocks=2, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, out_channels, blocks, stride):
            layers = []

            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )

            self.in_channels = out_channels

            for _ in range(1, blocks):
                layers.append(
                    BasicBlock(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=1
                    )
                )

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x
    ```




<quiz>
Conexões residuais ajudam principalmente a:
- [ ] Diminuir o uso de GPU
- [x] Facilitar fluxo de gradiente em redes profundas
- [ ] Remover necessidade de normalização
- [ ] Eliminar funções de ativação

O atalho preserva sinais e gradientes, mitigando o problema de degradação.
</quiz>






