# Detecção de Objetos: imagens, vídeo e contagem

Até agora, nossos modelos respondiam **qual é o objeto principal da imagem?** Nesta aula, a pergunta muda: **quais objetos aparecem e onde está cada um?**

[Lab 14 — Código](lab14/lab14.zip){ .md-button .md-button--primary }

[Abrir no Google Colab](https://colab.research.google.com/github/arnaldojr/computervision/blob/main/docs/aulas/lab14/notebook_deteccao_contagem.ipynb){ .md-button .md-button--primary }

[Baixar notebook](lab14/notebook_deteccao_contagem.ipynb){ .md-button download="notebook_deteccao_contagem.ipynb" }

[Baixar Video1](lab14/pessoa_estacao.mp4){ .md-button download="pessoa_estacao.mp4" }

[Baixar Video2](lab14/pessoa_rua.mp4){ .md-button download="pessoa_rua.mp4" }
### Execução local em tempo real

Para acompanhar o vídeo quadro a quadro em uma janela do OpenCV, use o script [contagem_local.py](lab14/contagem_local.py). No terminal, dentro da pasta `docs/aulas/lab14`:

```bash
pip install -r requirements.txt
python contagem_local.py
```

O script abre o vídeo padrão da pasta e exibe caixas, IDs, linha de contagem e total em tempo real. Durante a execução:

- `Espaço`: pausa ou continua o vídeo;
- `R`: reinicia a contagem;
- `Q` ou `Esc`: fecha a janela.

Para experimentar, altere no início de `contagem_local.py` as configurações `VIDEO_PATH`, `TARGET_CLASS`, `CONFIDENCE_THRESHOLD` e `LINE_RATIO`. Para usar a webcam, altere `USE_WEBCAM` para `True`.

---

## 1. Da classificação para a detecção

Uma classificação produz uma previsão por imagem:

```text
imagem de uma rua -> "carro" (91%)
```

Mas uma rua pode conter carros, pessoas, bicicletas e placas ao mesmo tempo. Um detector devolve uma lista de previsões:

```text
imagem de uma rua ->
  pessoa   92%  [x1, y1, x2, y2]
  carro    88%  [x1, y1, x2, y2]
  mochila  61%  [x1, y1, x2, y2]
```

Cada previsão tem três partes:

| Parte | Pergunta respondida | Exemplo |
|---|---|---|
| Classe | O que foi encontrado? | `pessoa` |
| Confiança | Quanto o modelo confia? | `0.92` |
| Bounding box | Onde está o objeto? | `[120, 80, 268, 220]` |

!!! question "Previsão"
    Uma foto tem três pessoas e dois carros. Quantas previsões deve retornar um classificador? E um detector? O detector pode retornar mais de cinco previsões se houver caixas duplicadas. Veremos por que isso acontece.

---

## 2. Bounding boxes: localizando o objeto

Uma **bounding box** é o menor retângulo que envolve um objeto. Um formato comum é:

$$
[x_{min}, y_{min}, x_{max}, y_{max}]
$$

- $(x_{min}, y_{min})$: canto superior esquerdo;
- $(x_{max}, y_{max})$: canto inferior direito.

A caixa também pode ser representada por centro, largura e altura:

$$
[x_c, y_c, w, h]
$$

Essa segunda forma é útil quando queremos acompanhar o centro de um objeto no vídeo.

---

## 3. Explore: IoU, confiança e NMS

Use o simulador abaixo. Arraste uma caixa prevista para aproximá-la ou afastá-la da anotação real. Em seguida, altere os limiares de confiança e IoU para observar quais previsões sobrevivem.

<div id="deteccao-widget" style="border: 1px solid #b7c6c2; padding: 16px; margin: 20px 0; background: #f8fbf9;" markdown="1">
<div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px;">
  <label>Confiança mínima <input data-confidence-threshold type="range" min="0" max="100" step="5" value="50"> <output data-confidence-value>50%</output></label>
  <label>IoU para NMS <input data-iou-threshold type="range" min="0" max="100" step="5" value="50"> <output data-iou-value>50%</output></label>
  <button type="button" data-reset-boxes>Reiniciar</button>
</div>
<canvas data-detection-canvas width="380" height="260" style="max-width: 100%; height: auto; border: 1px solid #b7c6c2; cursor: grab;" aria-label="Simulador de caixas delimitadoras"></canvas>
<div style="display: grid; gap: 5px; margin-top: 10px;">
  <span>IoU entre a melhor previsão e a anotação: <strong data-overlap-value></strong></span>
  <span data-nms-result></span>
</div>
</div>

### IoU: o quanto duas caixas coincidem?

O **Intersection over Union** compara a área comum entre duas caixas com a área total ocupada por elas:

$$
IoU = \frac{\text{área da interseção}}{\text{área da união}}
$$

- $IoU = 0$: as caixas não se tocam;
- $IoU = 1$: as caixas são idênticas;
- quanto maior o IoU, melhor a localização prevista.

A caixa tracejada no simulador é uma anotação humana, também chamada de **ground truth**. As caixas coloridas são hipóteses do detector.

### Confiança e NMS: por que caixas desaparecem?

O detector frequentemente propõe mais de uma caixa para o mesmo objeto. Primeiro, removemos previsões abaixo de uma confiança mínima. Depois, o **Non-Maximum Suppression (NMS)** mantém a caixa mais confiante e remove caixas muito parecidas com ela.

1. ordene as caixas por confiança;
2. mantenha a melhor;
3. calcule o IoU das demais com ela;
4. remova caixas cujo IoU esteja acima do limiar;
5. repita para as caixas restantes.

!!! tip "Experimento"
    Com confiança mínima em `50%`, diminua o limiar de IoU. Em que ponto a segunda caixa da pessoa deixa de aparecer? Agora aumente a confiança mínima para `80%`: o que acontece com a previsão de mochila?

---

## 4. O pipeline de detecção

Um detector pré-treinado recebe uma imagem e executa este fluxo:

```text
imagem -> pré-processamento -> modelo -> caixas candidatas
       -> filtro de confiança -> NMS -> caixas, classes e confianças finais
```

Na prática, começaremos com um detector YOLO pré-treinado no conjunto COCO. O COCO contém classes cotidianas como `person`, `car`, `bicycle`, `dog` e `bottle`.

O objetivo não é decorar uma API. Ao receber a saída do modelo, você deve saber ler:

```python
for box, confidence, class_id in detections:
    # box: onde o objeto está
    # confidence: quanto o detector confia
    # class_id: qual classe foi reconhecida
    ...
```

---

## 5. Detecção em vídeo

Um vídeo é uma sequência de imagens. O detector olha cada quadro separadamente:

```text
quadro 1 -> caixas e classes
quadro 2 -> caixas e classes
quadro 3 -> caixas e classes
```

Isso produz um vídeo anotado, mas ainda não resolve contagem. Se uma pessoa aparece por 100 quadros, somar as detecções produziria aproximadamente 100 pessoas, embora exista apenas uma.

!!! warning "Detecção não é rastreamento"
    **Detecção** responde: "o que está neste quadro?"

    **Rastreamento** responde: "esta caixa no quadro atual representa o mesmo objeto que apareceu antes?"

---

## 6. Rastreamento: dando identidade às caixas

Em detecção pura, cada quadro é independente. Mesmo que uma pessoa permaneça visível por vários quadros, o detector devolve apenas uma nova caixa para ela a cada vez:

```python
# Detecção: caixas, classes e confianças do quadro atual.
result = model(frame, conf=0.40)[0]
boxes = result.boxes.xyxy
```

Para acompanhar objetos no tempo, usamos um **rastreador**. Ele recebe as detecções de vários quadros e tenta associar cada caixa nova a uma caixa anterior. Quando a associação funciona, o objeto recebe um ID persistente:

```python
# Rastreamento: além das caixas, cada objeto recebe um ID.
tracked = model.track(frame, persist=True, conf=0.40)[0]
track_ids = tracked.boxes.id
```

Imagine uma pessoa atravessando cinco quadros:

```text
quadro 1: pessoa, caixa A, ID 7
quadro 2: pessoa, caixa B, ID 7
quadro 3: pessoa, caixa C, ID 7
```

As caixas mudam porque a pessoa se move, mas o ID tenta permanecer `7`. É isso que permite dizer que as três detecções representam uma pessoa, e não três pessoas diferentes.

### O que `persist=True` faz?

`persist=True` mantém a memória do rastreador entre uma chamada e a próxima. Sem essa memória, o rastreador começaria do zero em cada quadro e os IDs não teriam utilidade para uma contagem.

!!! question "Observe no laboratório"
  Antes de contar qualquer objeto, execute a etapa de rastreamento do notebook. Observe um objeto por alguns segundos e responda: ele mantém o mesmo ID? Em que situação o ID pode desaparecer ou mudar?

!!! warning "ID não é uma garantia"
  O rastreador pode perder ou trocar um ID quando o objeto fica oculto, sai da imagem, aparece muito pequeno ou se sobrepõe a outro objeto. Por isso, um contador baseado em tracking também pode errar.

---

## 7. Contagem por cruzamento de linha

Nossa primeira aplicação usa uma regra espacial simples:

1. desenhe uma linha virtual no vídeo;
2. calcule o centro de cada bounding box;
3. conte quando o centro cruza a linha;
4. registre o evento para não contar novamente imediatamente.

Para uma caixa $[x_{min}, y_{min}, x_{max}, y_{max}]$, seu centro é:

$$
x_c = \frac{x_{min} + x_{max}}{2}, \qquad
y_c = \frac{y_{min} + y_{max}}{2}
$$

Uma regra inicial para uma linha horizontal em $y = L$ é:

```python
cruzou_para_baixo = centro_anterior_y < L and centro_atual_y >= L
```

Agora podemos usar o ID persistente. Para cada ID, guardamos sua posição anterior e registramos a contagem apenas na primeira vez em que ele cruza a linha. Sem tracking, não saberíamos se o centro atual pertence ao mesmo objeto do quadro anterior.

