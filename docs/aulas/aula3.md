# Aula 3 — Filtros de Convolução: blur, realce, bordas, limiarização e blending

Nesta aula você vai entender (de verdade) o que é **um filtro de convolução** e por que quase toda pipeline de Visão Computacional começa com alguma forma de **suavização**, **realce** ou **detecção de bordas**.

[Lab03 — Filtros de Convolução (Notebook)](lab04/arquivolab4.zip){ .md-button .md-button-primary }

A meta aqui não é “decorar kernels”, e sim enxergar o padrão: **kernel + varredura + regra local** → efeito global na imagem.

---

## Objetivos de aprendizagem

Ao final da aula, você deve ser capaz de:

1. Explicar o processo **genérico** de filtragem por kernel no domínio espacial.
2. Diferenciar **convolução** e **correlação** (e saber quando isso importa).
3. Aplicar e comparar filtros de **blurring** (média, gaussiano).
4. Aplicar filtros de **sharpening** (realce) e entender o papel de valores negativos no kernel.
5. Usar detectores de bordas (Sobel/Laplaciano) e o **Canny** (com ajuste de limiares).
6. Aplicar **limiarização** (threshold) e interpretar seus modos.
7. Fazer **blending** (sobreposição) e combinar com filtros.
8. Processar **vídeo frame a frame** aplicando convolução em tempo real.

---

## 1) Pré-requisitos mínimos (para não errar “besteira”)

### 1.1 Imagem como matriz (revisão rápida)
- Imagem em tons de cinza: `img.shape == (H, W)`
- Imagem colorida (OpenCV): `img.shape == (H, W, 3)` e **ordem BGR** (não RGB)

!!! warning "OpenCV usa BGR"
    Se você pegar um kernel “por canal” ou fizer blending de imagens coloridas, lembre que o OpenCV lê como **BGR** por padrão.

### 1.2 `dtype` e saturação (uint8)
Filtros podem gerar valores **negativos** ou **maiores que 255**. Em `uint8`, isso pode “estourar” e causar resultado estranho.

!!! tip "Regra de ouro para depurar"
    Durante testes, converta temporariamente para `float32`, aplique o filtro e **normalize** ou faça `clip` antes de voltar para `uint8`.

---

## 2) A ideia central: kernel + vizinhança + soma ponderada

Um **kernel** (ou máscara) é uma pequena matriz (3×3, 5×5, 7×7…) que define como cada pixel será recalculado com base na sua **vizinhança**.

Intuição:
- O kernel “passa por baixo” da imagem.
- Em cada posição, você faz uma **soma ponderada** (produto elemento a elemento + soma).
- O resultado vira o novo valor do pixel central.

!!! info "Convolução vs correlação (o detalhe que confunde)"
    - **Correlação**: aplica o kernel “como ele é”.
    - **Convolução**: aplica o kernel **invertido** (espelhado).
    
    Na prática, em PDI muitos kernels são **simétricos**, então o resultado é o mesmo e as bibliotecas costumam implementar o comportamento como correlação.

---

## 3) Parte A — Filtros de Blurring (suavização)

**O que faz:** reduz ruído, “alisa” a imagem, diminui detalhes finos.

### 3.1 Média (box filter)
Kernel típico 3×3:
\[
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1
\end{bmatrix}
\]

**Intuição:** todo mundo pesa igual.

### 3.2 Gaussiano
Parecido com média, mas o centro pesa mais.

**Intuição:** suaviza sem “destruir” tanto as bordas quanto o box filter.

!!! tip "Tamanho do kernel importa"
- Kernel maior → blur mais forte (mais perda de detalhe)
- Kernel menor → blur mais leve

#### Quiz (rápido)
<quiz>
<question>
Em um kernel de suavização “bem comportado”, a soma dos coeficientes deve ser:
</question>
<answer type="A">0</answer>
<answer type="B" correct="true">1 (ou normalizada para 1)</answer>
<answer type="C">255</answer>
<answer type="D">Depende do formato da imagem</answer>
</quiz>

### 3.3 Exercícios do notebook
#### Desafio 1 — Estudo de blurring
- Escolha uma imagem.
- Compare filtros de borramento e **varie o tamanho do kernel**.
- Escreva 2–3 linhas do que você observou (detalhe, ruído, bordas).

---

## 4) Parte B — Sharpening (realce)

**O que faz:** enfatiza detalhes e bordas (aumenta contraste local).

Um kernel clássico 3×3 (exemplo):
\[
\begin{bmatrix}
0 & -1 & 0\\
-1 & 5 & -1\\
0 & -1 & 0
\end{bmatrix}
\]

**Intuição:** você “pune” vizinhos e “recompensa” o centro.

!!! warning "Sharpening amplifica ruído"
Se a imagem já tiver ruído, sharpening pode piorar. Muitas pipelines fazem:
1) blur leve → 2) sharpening → 3) pós-processamento

#### Quiz (rápido)
<quiz>
<question>
Por que kernels de realce costumam ter valores negativos?
</question>
<answer type="A">Para reduzir o tamanho da imagem</answer>
<answer type="B">Porque OpenCV exige valores negativos</answer>
<answer type="C" correct="true">Para subtrair contribuição de vizinhos e aumentar contraste local</answer>
<answer type="D">Para converter BGR em RGB</answer>
</quiz>

### 4.1 Exercícios do notebook
#### Desafio 2 — Estudo de contraste/realce
- Escolha uma imagem.
- Teste kernels de realce (sharpen) e **varie o tamanho do kernel** quando aplicável.
- Compare: “ficou mais nítido” vs “ficou mais ruidoso”.

---

## 5) Parte C — Detecção de bordas e o Canny

### 5.1 Bordas por derivadas (Sobel/Laplaciano)
- **Sobel**: aproxima derivadas em X e Y → destaca transições.
- **Laplaciano**: segunda derivada → destaca regiões onde a intensidade muda rápido.

!!! tip "Quase sempre: blur antes de borda"
Uma suavização leve antes do detector tende a estabilizar bordas e reduzir falsos positivos por ruído.

### 5.2 Canny (robusto e muito usado)
O Canny é um pipeline:
1) (normalmente) blur
2) gradiente
3) supressão de não-máximos
4) histerese com **dois limiares** (`threshold1`, `threshold2`)

**Intuição dos limiares:**
- `threshold2` (alto): bordas “fortes”
- `threshold1` (baixo): bordas “fracas” que só entram se conectadas a uma forte

#### Quiz (rápido)
<quiz>
<question>
No Canny, o que tende a acontecer se você diminuir bastante os dois thresholds?
</question>
<answer type="A">Menos bordas</answer>
<answer type="B" correct="true">Mais bordas e mais ruído/“sujeira”</answer>
<answer type="C">A imagem fica colorida</answer>
<answer type="D">O kernel muda automaticamente</answer>
</quiz>

### 5.3 Exercícios do notebook
#### Desafio 3 — Ajuste de thresholds no Canny
- Teste diferentes imagens.
- Ajuste `threshold1` e `threshold2`.
- Procure um “equilíbrio” entre:
  - detectar bordas reais
  - evitar ruído e textura irrelevante

---

## 6) Parte D — FILTRO DE LIMIARIZAÇÃO (threshold)

Limiarização converte tons de cinza em uma imagem binária (ou quase binária), classificando pixels com base em um limiar.

Principais modos do OpenCV (ideia geral):
- `THRESH_BINARY`: acima do limiar → 255; abaixo → 0
- `THRESH_BINARY_INV`: invertido do binary
- `THRESH_TRUNC`: acima → vira o limiar; abaixo mantém
- `THRESH_TOZERO`: abaixo → 0; acima mantém
- `THRESH_TOZERO_INV`: acima → 0; abaixo mantém

!!! tip "Quando usar limiarização?"
- Objetos bem destacados do fundo (diferença de intensidade clara)
- Pré-processamento para contornos, OCR simples, segmentação inicial

#### Quiz (rápido)
<quiz>
<question>
Qual modo é o mais “clássico” para transformar uma imagem em preto e branco puro (binária)?
</question>
<answer type="A" correct="true">THRESH_BINARY</answer>
<answer type="B">THRESH_TRUNC</answer>
<answer type="C">THRESH_TOZERO</answer>
<answer type="D">THRESH_TOZERO_INV</answer>
</quiz>

---

## 7) Parte E — Sobreposição de imagens (blending)

O blending combina duas imagens como uma média ponderada:

\[
g(x) = (1-\alpha)\,f_0(x) + \alpha\,f_1(x)
\]

- `α` perto de 0 → aparece mais `f0`
- `α` perto de 1 → aparece mais `f1`

!!! warning "Pré-condição para blending"
As duas imagens precisam ter **mesmo tamanho** (H×W) e o mesmo número de canais, ou você deve ajustar (resize/crop).

### 7.1 Exercícios do notebook
#### Desafio 4 — Blending + filtros
- Faça blending.
- Aplique **antes ou depois** um blur ou realce.
- Compare: “blending → filtro” vs “filtro → blending”.

---

## 8) Desafio final — Vídeo em tempo real (portfólio)

#### Desafio 5 — Script `.py` que processa vídeo (webcam ou `.mp4`)
Você vai construir um script que:
1) captura frames (`cv2.VideoCapture`)
2) aplica um filtro por convolução em **cada frame**
3) exibe o resultado em tempo real (`cv2.imshow`)

**Requisitos mínimos**
- Escolha e implemente pelo menos **uma máscara**:
  - bordas (Sobel/Laplaciano)
  - blur (média/gaussiano)
- Explique no código (comentário curto) o efeito da máscara.
- Organize o código em **funções** (responsabilidades claras).

!!! tip "Checklist de qualidade"
- `main()` claro
- função `apply_filter(frame)` (ou similar)
- tratamento de tecla para sair (`q`/`esc`)
- se usar webcam: cheque se abriu corretamente

---

## 9) Fechamento e ponte para a próxima aula

Hoje você trabalhou com filtros no domínio espacial (kernels). Na próxima etapa, faz sentido avançar para:
- segmentações mais robustas (HSV e `inRange`)
- morfologia (erosão/dilatação)
- contornos e extração de forma
- e, mais à frente, features e descritores

A diferença entre “brincar com kernels” e “usar kernels profissionalmente” é: **saber por que você está filtrando** (ruído? borda? realce?) e **conseguir justificar os parâmetros** (kernel, sigma, thresholds).

