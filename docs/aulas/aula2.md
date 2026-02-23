# Aula 2 — Operações básicas: negativo, recorte (ROI) e segmentação simples

Nesta aula você vai aplicar **operações fundamentais** de manipulação de imagem (pixel a pixel) que aparecem o tempo todo em Visão Computacional.

[Lab02 — Manipulação básica (Notebook)](lab02/arquivolab2.zip){ .md-button .md-button-primary }

A ideia é entender **o que está acontecendo com os valores dos pixels** — sem “pular direto” para funções prontas.

---

## Objetivos de aprendizagem

Ao final desta aula, você deve ser capaz de:

1. Implementar **negativo** (inversão de intensidade) em imagem **em tons de cinza** e em **RGB**.
2. Aplicar **recorte (crop)** para extrair uma **ROI** usando slicing (`img[y1:y2, x1:x2]`).
3. Realizar uma **segmentação simples** baseada em um **limiar** (threshold) em um canal (ex.: canal **G**).
4. Explicar limitações de operações “na mão” e por que técnicas mais robustas aparecem na próxima aula.

---

## 1) Pré-requisitos mínimos (para não errar “besteira”)

### 1.1 Ordem de indexação (y antes de x)

Em NumPy/OpenCV, o acesso é sempre:

- `img[y, x]` em imagem 2D (tons de cinza)
- `img[y, x, c]` em imagem 3D (colorida)

!!! warning "Erro clássico"
    Trocar `(x, y)` por `(y, x)` **recorta a área errada** e pode parecer que “o código está bugado”.

<quiz>
Em `img[y, x]`, qual índice representa a **linha** (altura)?
- [x] `y`
- [ ] `x`
- [ ] depende do formato da imagem

Trocar `(x, y)` por `(y, x)` **recorta a área errada** e pode parecer que “o código está bugado”.
</quiz>

### 1.2 `dtype` e range (por que 255 aparece sempre)

A maioria dos exemplos do notebook usa imagens `uint8`:

- valores inteiros no intervalo **0..255**
- 0 = preto, 255 = branco (em tons de cinza)

<quiz>
Se a imagem é `uint8`, qual é o valor máximo que um pixel pode ter?
- [x] 255
- [ ] 1
- [ ] 1024
</quiz>

---

## 2) Parte A — Filtro negativo (inversão)

### 2.1 Intuição

O **negativo** troca “claro ↔ escuro”.

Em uma imagem `uint8`, a inversão explícita é:

- `novo = 255 - antigo`

Isso vale para:
- **tons de cinza** (um valor por pixel)
- **RGB** (três valores por pixel, um por canal)

!!! tip "Como conferir se está certo"
    Pegue um pixel e faça na mão:
    - se `antigo = 10`, então `novo = 245` (quase branco)
    - se `antigo = 200`, então `novo = 55` (escurece)

<quiz>
Se um pixel em tons de cinza vale 80, qual será o valor no negativo (`uint8`)?
- [x] 175
- [ ] 80
- [ ] 255
</quiz>

### 2.2 Tons de cinza vs RGB

- Em **tons de cinza**, você inverte um único valor.
- Em **RGB**, você inverte **cada canal**.

Exemplo conceitual (um pixel):
- antes: `(R,G,B) = (20, 120, 200)`
- depois: `(235, 135, 55)`

<quiz>
Em uma imagem RGB, o negativo é aplicado:
- [x] em cada canal (R, G e B)
- [ ] apenas no canal R
- [ ] apenas no canal com maior intensidade
</quiz>

### 2.3 Exercícios do notebook

#### Desafio 1 — Negativo em escala de cinza (0..255)

Implemente uma função/trecho que inverte uma imagem em tons de cinza **pixel a pixel**.

!!! note "Dica do próprio notebook"
    A inversão explícita é: `a = 255 - a`

#### Desafio 2 — Negativo em imagem colorida (RGB)

Faça o mesmo para uma imagem colorida, invertendo **R, G e B**.

!!! tip "Validação rápida"
    Se a imagem original tem muito céu azul, o negativo tende a puxar para tons “amarelados/avermelhados”.

---

## 3) Parte B — Recorte de imagem (Crop / ROI)

### 3.1 O que é ROI?

ROI (*Region of Interest*) é uma sub-região da imagem que você quer analisar.

Recorte por slicing:

- `roi = img[y1:y2, x1:x2]`

!!! warning "Ordem importa"
    `img[y1:y2, x1:x2]` (linhas primeiro).  
    Se você inverter, você recorta outra região.

<quiz>
Qual recorte extrai uma ROI do canto superior esquerdo de tamanho 100×200 (H×W)?
- [x] `img[0:100, 0:200]`
- [ ] `img[0:200, 0:100]`
- [ ] `img[0:100, 0:200, 0]`
</quiz>

### 3.2 Dica prática: sempre cheque o `shape`

Depois do recorte, confira:

- `roi.shape == (y2 - y1, x2 - x1)` (em tons de cinza)
- `roi.shape == (y2 - y1, x2 - x1, 3)` (em RGB)

<quiz>
Se `roi = img[50:250, 580:950]` (imagem RGB), qual é a altura da ROI?
- [x] 200
- [ ] 370
- [ ] 950
</quiz>

### 3.3 Exercício do notebook

#### Desafio 3 — “Ajude o sayajin!!”

Você deve recortar a região correta da imagem, criando uma ROI que resolve o problema proposto no notebook.

!!! tip "Como não se perder"
    1. Comece com um recorte grande (garanta que pega a área).
    2. Vá ajustando `x1, x2, y1, y2` até “encaixar”.
    3. Se ficar estranho, você provavelmente trocou x e y.

---

## 4) Parte C — Segmentação simples por limiar (threshold “na mão”)

### 4.1 Ideia: separar “algo” do fundo

Segmentar é criar uma regra do tipo:

- **se** o pixel satisfaz uma condição → pinta de uma cor (ex.: branco)
- **senão** → pinta de outra (ex.: preto)

No notebook, aparece uma ideia simples (didática):

- olhar o **canal G** (`img[y, x, 1]`)  
- usar um limiar (ex.: `> 170`)

!!! note "Por que canal G?"
    Não é “a melhor” forma — é só um jeito **simples** de começar a enxergar como condições em pixels viram uma segmentação.

<quiz>
Se você aumentar o limiar de 170 para 220, a segmentação tende a:
- [x] selecionar menos pixels (fica mais restrita)
- [ ] selecionar mais pixels (fica mais ampla)
- [ ] não mudar nada
</quiz>

### 4.2 Limitações (importante!)

Esse tipo de regra:
- depende muito da iluminação
- falha com sombras/reflexos
- não generaliza bem para outras cenas

Isso é **intencional**: serve para você entender o mecanismo básico.  
Na próxima aula, a segmentação vai ficar mais robusta.

### 4.3 Exercícios do notebook

#### Desafio 4 — Experimente (aperitivo)

Modifique parâmetros e lógica do código para observar:
- como o limiar altera o resultado
- como escolher canal (R/G/B) muda tudo
- como pequenos ajustes “quebram” ou “melhoram” a segmentação

!!! tip "Sugestões de variação"
    - troque `> 170` por `> 150` e `> 200`
    - tente o canal `0` (R) e o canal `2` (B)
    - em vez de pintar branco, pinte de vermelho `(255,0,0)` para visualizar melhor

#### Desafio 5 — Faça o contrário: “fundo sem o drone”

A ideia é inverter a regra:
- em vez de destacar o “fundo”, tente remover o objeto (ou vice-versa)

