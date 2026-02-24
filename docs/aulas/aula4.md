# Aula 4 — Espaço de cor (HSV) e Contornos: segmentação robusta e extração de forma

Nesta aula você vai sair do “threshold ingênuo em RGB” e passar para uma segmentação mais robusta usando **HSV**,
e depois vai aprender a extrair **contornos** e medir propriedades geométricas (área, perímetro, caixas, centróide).
Esse é um bloco essencial para projetos de detecção simples, rastreamento por cor e visão aplicada a robótica/IoT.

[Lab04 — Espaço de cor e Contornos (Notebook)](lab05/arquivolab5.zip){ .md-button .md-button-primary }

---

## Como usar este handout

- **Antes do notebook:** leia as seções 1 e 2 para não confundir HSV/BGR e nem “se perder” em máscaras.
- **Durante:** siga a sequência: *converter → segmentar (inRange) → limpar (morfologia) → contornos → métricas*.
- **Depois:** faça o Desafio Final (rastreamento completo por cor + forma). Ele vira mini-projeto.

---

## Objetivos de aprendizagem

Ao final da aula, você deve ser capaz de:

1. Explicar por que **HSV** é mais adequado do que RGB/BGR para segmentação por cor.
2. Converter imagens BGR → HSV e escolher faixas `(H,S,V)` coerentes.
3. Criar máscaras com `inRange()` e aplicar a máscara na imagem original.
4. Usar **operações morfológicas** (erosão/dilatação/open/close) para remover ruído e preencher falhas.
5. Extrair **contornos** com `findContours()` e desenhá-los corretamente.
6. Medir propriedades de contornos: **área, perímetro, centróide, bounding box**.
7. Filtrar contornos por critérios (ex.: área mínima, proporção, circularidade).
8. Construir um pipeline simples de **detecção e marcação** de objetos por cor.

---

## 1) Pré-requisitos mínimos

### 1.1 OpenCV lê em BGR
Ao carregar imagens com OpenCV, a ordem é **BGR**. Você converte para HSV antes de segmentar.

!!! warning "Erro clássico"
Segmentation por “RGB” sem converter e sem considerar iluminação geralmente falha. Em OpenCV, primeiro lembre do BGR e só depois pense em HSV.

### 1.2 Máscara é uma imagem (quase sempre) binária
Uma máscara é uma matriz `H×W` com valores 0/255 (ou 0/1). Ela serve para selecionar pixels.

!!! tip "Como depurar rápido"
    Mostre sempre:
    1) a imagem original  
    2) a imagem em HSV (às vezes vale visualizar canais)  
    3) a máscara resultante  
    4) o resultado final (`bitwise_and`)

---

## 2) Por que HSV ajuda na segmentação?

No RGB/BGR, cor e iluminação estão “misturados”. No HSV, você separa:

- **H (Hue / Matiz):** “qual cor é” (vermelho, verde, azul…)
- **S (Saturation / Saturação):** “quão forte” é a cor
- **V (Value / Valor):** brilho/iluminação

**Intuição prática:** em muitos cenários, o objeto continua “verde” mesmo com variação de luz,
mas seu canal V muda — por isso HSV tende a ser mais estável para segmentar.

#### Quiz (rápido)
<quiz>
Qual componente do HSV está mais relacionado à variação de iluminação/brilho?
- [ ] H
- [ ] S
- [x] V
- [ ] Nenhum

</quiz>

---

## 3) Parte A — Segmentação por cor com `inRange()` (o coração da aula)

### 3.1 Pipeline mínimo
1) `cvtColor(BGR → HSV)`
2) `mask = inRange(hsv, lower, upper)`
3) `result = bitwise_and(img, img, mask=mask)`

!!! tip "Faixas (lower/upper) são hipóteses"
Os limites de HSV não são “universais”. Você ajusta conforme:
- iluminação do ambiente
- câmera
- material do objeto
- sombras / reflexos

### 3.2 O detalhe do Hue (vermelho é especial)
Em HSV, o Hue é circular (0 “encosta” em 179 no OpenCV). O vermelho costuma exigir duas faixas:
- faixa baixa (próximo de 0)
- faixa alta (próximo de 179)

!!! info "Se o notebook usar vermelho"
Se você segmentar vermelho e a máscara falhar, suspeite do “wrap-around” do Hue e teste duas máscaras + OR.

#### Exercício 1 — Criar máscara de uma cor
- Escolha uma cor-alvo no notebook (ex.: verde/azul).
- Encontre limites `lower` e `upper`.
- Mostre máscara e resultado aplicado.

#### Quiz (rápido)
<quiz>
Se você aumentar muito o limiar mínimo de Saturação (S) na faixa HSV, o que tende a acontecer?
- [ ] Mais pixels entram na máscara
- [x] Menos pixels entram (cores “fracas”/acinzentadas saem)
- [ ] A imagem fica mais clara
- [ ] A máscara vira RGB

</quiz>

---

## 4) Parte B — Limpeza de máscara: morfologia (quando a máscara fica “suja”)

Segmentação real quase sempre dá:
- pontos isolados (ruído)
- buracos no objeto

Operações morfológicas típicas:
- **Erosão:** “come” pixels (remove ruído pequeno, afina)
- **Dilatação:** “engorda” pixels (fecha falhas pequenas)
- **Abertura (open):** erosão → dilatação (remove ruído)
- **Fechamento (close):** dilatação → erosão (fecha buracos)

!!! tip "Regra prática"
- Máscara com *pontos soltos* → **open**
- Máscara com *buracos no objeto* → **close**

#### Exercício 2 — Open vs Close
- Aplique `open` e `close` na máscara.
- Compare lado a lado e descreva o efeito.

---

## 5) Parte C — Contornos: do binário para a geometria

### 5.1 O que é um contorno?
É uma curva (lista de pontos) que delimita o “borda” de um componente conectado na máscara.

Pipeline:
1) ter uma máscara binária bem definida (idealmente limpa)
2) `findContours(mask, ...)`
3) `drawContours(...)` na imagem

!!! warning "findContours altera a imagem em algumas versões"
Em alguns usos, é comum passar `mask.copy()` para evitar efeitos colaterais.

### 5.2 Métricas básicas
Com o contorno em mãos, você consegue:
- **Área:** `contourArea(cnt)`
- **Perímetro:** `arcLength(cnt, True)`
- **Bounding box:** `boundingRect(cnt)` (retângulo alinhado aos eixos)
- **Centróide:** via momentos (`moments`)

!!! tip "Centróide robusto"
Para centróide, use momentos:
- `M = moments(cnt)`
- `cx = M["m10"]/M["m00"]`, `cy = M["m01"]/M["m00"]`  
e sempre cheque `m00 != 0`.

#### Quiz (rápido)
<quiz>
Por que é comum filtrar contornos por área mínima antes de desenhar/analisar?
- [ ] Para converter HSV em BGR
- [ ] Para acelerar a câmera
- [x] Para ignorar ruído (pequenos blobs) e focar em objetos relevantes
- [ ] Porque OpenCV exige

</quiz>


---

## 6) Parte D — Filtrando “o objeto certo” (critério de forma)

Depois de obter vários contornos, você escolhe o “alvo” usando critérios:

### 6.1 Critério por área (o mais comum)
- `area > area_min`

### 6.2 Proporção (aspect ratio) usando bounding box
- `w/h` para evitar selecionar objetos “muito achatados” ou “muito altos”.

### 6.3 Circularidade (quando o alvo é mais “redondo”)
Uma medida simples:
\[
c = \frac{4\pi \cdot area}{perimetro^2}
\]
- perto de 1 → círculo
- menor → formas irregulares

!!! tip "Critérios combinados"
Em visão aplicada, a robustez vem de combinar:
**cor (HSV) + limpeza (morfologia) + forma (área/aspect/circularidade)**.

---

## 7) Exercícios guiados (do notebook)

### Desafio 3 — Pipeline completo (imagem)
- Converter BGR→HSV
- `inRange` para segmentar cor
- Open/Close para limpar
- Encontrar contornos
- Desenhar o contorno do maior objeto
- Mostrar bounding box + centróide

### Desafio 4 — Ajuste de robustez
- Crie 2 cenários (ex.: luz mais forte e mais fraca)
- Ajuste limites HSV e morfologia para manter o resultado estável
- Documente suas escolhas (2–4 linhas)

---

## 8) Desafio Final — Rastreamento por cor + forma (mini-projeto)

Você vai construir um script `.py` (fora do notebook) que:

1) Captura webcam (`VideoCapture`)
2) Converte frame para HSV
3) Segmenta uma cor alvo (com limites ajustáveis)
4) Limpa máscara com morfologia
5) Extrai contornos
6) Seleciona o objeto-alvo (ex.: maior por área ou por circularidade)
7) Desenha:
   - contorno
   - bounding box
   - centróide
   - texto com área e posição

**Requisitos mínimos**
- Código organizado em funções (`segment()`, `clean_mask()`, `find_target()`, `draw_overlay()`)
- Tratamento de saída (`q` para sair)
- Comentários curtos justificando:
  - limites HSV escolhidos
  - morfologia usada (open/close e kernel)

!!! tip "Upgrade opcional (vale bônus)"
- Permitir calibrar HSV com trackbars
- Exibir FPS
- Salvar um frame quando apertar uma tecla

---

## 9) Fechamento e ponte para a próxima aula

Hoje você aprendeu a base de:
- **segmentação robusta por cor** (HSV)
- **extração de forma** (contornos e métricas)

Isso abre caminho para:
- contornos avançados (hull, aproximação poligonal)
- detecção por forma (triângulo, retângulo, círculo)
- tracking (centroide ao longo do tempo)
- integração com projetos (IoT/robótica, inspeção visual, interfaces)

A diferença entre uma demo e uma solução real é: **conseguir justificar parâmetros** e **tornar o pipeline estável** sob variação de cenário.

