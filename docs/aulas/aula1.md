# Aula 1 — Fundamentos de Imagem Digital e Representação em Memória (PDI)

Nesta aula você vai consolidar **a teoria mínima** necessária para executar o notebook prático:

[Lab01 — Intro PID](lab01/arquivolab1.zip){ .md-button .md-button-primary }

O objetivo não é decorar termos, mas entender **como a imagem é representada** e **o que acontece quando você lê, visualiza e altera pixels**.

---

## Como usar este handout

!!! tip "Fluxo recomendado"
    1. **Antes do notebook:** leia até a seção 5 e responda os quizzes conforme aparecem.  
    2. **Durante o notebook:** use este material como referência (principalmente BGR/RGB, `shape`, `dtype`, `resize`).  
    3. **Depois (5 min):** revise apenas as perguntas que errou e valide no código.

---

## Objetivos de aprendizagem

Ao final desta aula, você deve ser capaz de:

1. Explicar o que significa uma imagem ser um **sinal discreto 2D** (amostragem + quantização).
2. Interpretar a estrutura de uma imagem em Python como **arrays NumPy** (`H×W×C`).
3. Diferenciar **BGR (OpenCV)** de **RGB (Matplotlib)** e evitar visualização “com cores erradas”.
4. Identificar `dtype` (ex.: `uint8`) e o **range de valores** que ele suporta.
5. Aplicar operações simples: **tons de cinza, resize/amostragem, leitura de pixel, alteração de pixels**.
6. Entender por que **varrer imagem com `for`** é lento e quando isso faz sentido.

---

## 1) Imagem digital: amostragem e quantização

Uma imagem digital pode ser vista como um sinal contínuo (mundo real) convertido em uma estrutura discreta:

- **Amostragem (sampling):** define **quantos pixels** teremos (resolução).
- **Quantização (quantization):** define **quantos valores possíveis** cada pixel pode assumir.

Em termos práticos:
- Mais resolução ⇒ mais detalhes espaciais, mas maior custo de processamento.
- Mais bits por pixel ⇒ mais níveis de intensidade (ex.: 8 bits ⇒ 256 níveis).

<quiz>
Qual definição descreve melhor **amostragem** em imagens?
- [x] Definir a quantidade de pixels (resolução) que representa a cena
- [ ] Definir quantos níveis de intensidade cada pixel pode assumir
- [ ] Definir a ordem dos canais de cor (RGB/BGR)

Amostragem está ligada à **densidade espacial** (quantos pontos/pixels).
</quiz>

<quiz>
Uma imagem de 8 bits por pixel tem quantos níveis possíveis de intensidade?
- [ ] 128
- [x] 256
- [ ] 1024

8 bits ⇒ 2⁸ = 256 valores.
</quiz>

---

## 2) Como a imagem aparece no código (NumPy)

No notebook, após carregar uma imagem, você verá algo como:

- **Imagem em tons de cinza:** array 2D  
  `shape = (H, W)`
- **Imagem colorida:** array 3D  
  `shape = (H, W, C)` com `C = 3` (canais)

**Interpretação:**
- `H` = número de **linhas** (altura)
- `W` = número de **colunas** (largura)
- `C` = **canais de cor**

!!! tip "Regra prática de indexação"
    Em NumPy/OpenCV você acessa como `img[y, x]` (linha primeiro, coluna depois).  
    Ou seja: **(y, x)** ≈ **(linha, coluna)**.

<quiz>
Em NumPy/OpenCV, se `img.shape == (720, 1280, 3)`, o que isso significa?
- [x] A imagem tem 720 linhas, 1280 colunas e 3 canais
- [ ] A imagem tem 1280 linhas, 720 colunas e 3 canais
- [ ] A imagem tem 720 linhas, 1280 colunas e 1280 canais

`shape = (H, W, C)` onde `H` é altura (linhas) e `W` é largura (colunas).
</quiz>

<quiz>
Qual acesso é correto para pegar um pixel na posição (x=50, y=10) em NumPy/OpenCV?
- [x] `img[10, 50]`
- [ ] `img[50, 10]`
- [ ] `img(x=50, y=10)`

A indexação é `img[y, x]` (linha, coluna).
</quiz>

---

## 3) BGR vs RGB: por que as cores “ficam erradas”?

- **OpenCV (`cv2`)** lê imagens em **BGR** (Blue, Green, Red).
- **Matplotlib (`plt.imshow`)** espera **RGB**.

Se você fizer:
```python
img = cv2.imread("NATUREZA_1.jpg")
plt.imshow(img)
```
as cores provavelmente ficarão trocadas.

A correção é converter:
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
```

!!! warning "Pegadinha clássica"
    Se você não converter **BGR→RGB**, seu pipeline pode “parecer” errado,
    mas o problema é **apenas de visualização** (não necessariamente do processamento).

<quiz>
Qual é a ordem padrão de canais ao carregar uma imagem com `cv2.imread()`?
- [x] BGR
- [ ] RGB
- [ ] HSV

OpenCV usa BGR por padrão.
</quiz>

<quiz>
Para visualizar corretamente no Matplotlib uma imagem lida pelo OpenCV, você normalmente faz:
- [x] Converter BGR → RGB com `cv2.cvtColor`
- [ ] Converter RGB → HSV com `cv2.cvtColor`
- [ ] Apenas usar `plt.imshow(img)` sem conversão

Matplotlib espera RGB.
</quiz>

---

## 4) Tons de cinza: o que muda?

Uma imagem em tons de cinza pode ser vista como **uma única banda** de intensidade.

No OpenCV:
- `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`  
- ou `cv2.imread(path, 0)`

Na visualização com Matplotlib, use:
```python
plt.imshow(img_gray, cmap="gray")
```

!!! note "Por que usar `cmap='gray'`?"
    Sem o `cmap`, o Matplotlib pode aplicar um mapa de cores (colormap) que não representa
    “cinza” de verdade, o que atrapalha sua interpretação.

<quiz>
No OpenCV, carregar uma imagem diretamente em tons de cinza pode ser feito com:
- [x] `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`
- [x] `cv2.imread(path, 0)`
- [ ] `cv2.imread(path, cv2.COLOR_BGR2GRAY)`

`cv2.COLOR_BGR2GRAY` é usado com `cv2.cvtColor`, não como flag do `imread`.
</quiz>

<quiz>
Ao usar `plt.imshow()` para mostrar uma imagem em tons de cinza, o mais correto é:
- [x] `plt.imshow(img_gray, cmap="gray")`
- [ ] `plt.imshow(img_gray)` (sem nada)
- [ ] `plt.imshow(img_gray, cmap="rgb")`

O `cmap="gray"` preserva a interpretação correta de intensidade.
</quiz>

---

## 5) Tipos (`dtype`) e range: por que isso importa?

A maioria das imagens lidas com OpenCV vem como:

- `dtype = uint8`
- valores no range **[0, 255]**

Isso tem consequências:
- Ao fazer contas, você pode ter **overflow** ou **clipping** se não controlar o tipo.
- Para operações matemáticas (ex.: normalização, filtros, gamma), é comum converter para `float32`,
  calcular e depois voltar para `uint8` com recorte.

Exemplo de boa prática (ideia geral):
```python
img_f = img.astype("float32") / 255.0
# ... processa ...
img_u8 = (img_f * 255).clip(0, 255).astype("uint8")
```

<quiz>
Uma imagem `uint8` tipicamente tem valores de intensidade entre [[0]] e [[255]].

---
`uint8` comporta 256 níveis. Se você fizer operações sem cuidado, pode ocorrer clipping/overflow.
</quiz>

<quiz>
Qual é o motivo mais comum para converter uma imagem para `float32` antes de certas operações?
- [x] Evitar problemas de overflow/clipping e trabalhar com valores normalizados
- [ ] Porque `uint8` não suporta imagens coloridas
- [ ] Porque `float32` é sempre mais rápido em todas as operações

Em processamento, `float32` ajuda a manter a matemática correta, principalmente ao normalizar.
</quiz>

---

## 6) Amostragem na prática: resize e interpolação

No notebook, você usa `cv2.resize()` para mudar o tamanho:

```python
img2 = cv2.resize(img_rgb, (600, 400), cv2.INTER_LINEAR)
```

Dois pontos importantes:
1. O tamanho é informado como **(largura, altura)** — atenção: é o inverso do `shape`.
2. A interpolação define como os novos pixels são estimados:
   - `INTER_NEAREST`: rápido, pode “pixelar”
   - `INTER_LINEAR`: padrão, bom para muitos casos
   - `INTER_AREA`: geralmente bom para reduzir (downsample)

<quiz>
O `cv2.resize(img, (600, 400), ...)` recebe o tamanho no formato:
- [x] (largura, altura)
- [ ] (altura, largura)
- [ ] (linhas, colunas, canais)

Atenção: isso difere do `shape`, que é (altura, largura, canais).
</quiz>

<quiz>
Qual interpolação tende a ser boa para reduzir a imagem (downsample)?
- [ ] `cv2.INTER_NEAREST`
- [ ] `cv2.INTER_LINEAR`
- [x] `cv2.INTER_AREA`

`INTER_AREA` costuma ser uma boa escolha ao diminuir imagens.
</quiz>

---

## 7) Acessando e alterando pixels

Acesso a um pixel (colorido):
```python
(b, g, r) = img_bgr[y, x]
```

Acesso a um pixel (cinza):
```python
v = img_gray[y, x]
```

Alterar pixels **pode** ser feito com `for`, mas isso é caro:

```python
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        img[y, x] = (255, 0, 0)
```

!!! warning "Custo computacional"
    Uma imagem 1080p tem ~2 milhões de pixels.  
    Dois `for` aninhados em Python ficam lentos rapidamente.

**Alternativa profissional:** usar operações vetorizadas (NumPy) e máscaras.
Você verá isso nas próximas aulas.

<quiz>
Marque as afirmativas verdadeiras sobre varrer uma imagem com `for` em Python:
- [x] Dois loops aninhados percorrem todos os pixels e podem ser lentos em imagens grandes
- [x] Em geral, operações vetorizadas com NumPy são mais eficientes que loops puros em Python
- [ ] Loops sempre são mais rápidos que operações vetorizadas
- [ ] Varrer a imagem não muda a complexidade do algoritmo

Loops em Python puro são didáticos, mas em produção evitamos quando possível.
</quiz>
