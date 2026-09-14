# Treinando um detector para o seu problema

Nas aulas anteriores, usamos um detector pronto e criamos um dataset anotado. Agora vamos conectar essas duas coisas: usar as imagens e as caixas que você produziu para ajustar um modelo YOLO ao seu problema.

[Abrir no Google Colab](https://colab.research.google.com/github/arnaldojr/computervision/blob/main/docs/aulas/lab16/yolo-treino.ipynb){ .md-button .md-button--primary }


[Baixar notebook](lab16/yolo-treino.ipynb){ .md-button download="yolo-treino.ipynb" }

## O que acontece nesta aula?

Ao final, você deve conseguir responder quatro perguntas:

1. Como o YOLO encontra as imagens, labels e classes do meu dataset?
2. O que estou escolhendo quando defino épocas, tamanho da imagem e batch?
3. Como saber se o treino melhorou de verdade?
4. Como testar o arquivo `best.pt` em uma imagem que o modelo não viu?

O caminho completo é este:

```text
dataset anotado -> data.yaml -> modelo pré-treinado -> treino -> validação -> best.pt -> inferência
```

---

## 1. Antes de apertar "treinar"

No Lab 13, cada imagem recebeu um arquivo `.txt` com as caixas e as classes dos objetos. Para treinar, a única exigência nova é separar parte dessas imagens para validação.

```text
meu-dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

As imagens em `train` são usadas para ajustar o modelo. As imagens em `val` ficam fora desse ajuste e servem para verificar se ele aprendeu algo que funciona além dos exemplos vistos.

!!! warning "Regra que evita resultados enganosos"
    Uma imagem não pode aparecer em `train` e `val`. Em vídeos, quadros muito próximos também não devem ser divididos entre os dois conjuntos: eles são quase a mesma imagem.

### Checagem rápida do dataset

Antes do treino, abra algumas imagens de cada pasta e confira os respectivos labels. Procure principalmente por:

- caixa na classe errada;
- caixa que não cobre o objeto;
- imagem sem o `.txt` correspondente;
- índice de classe fora da lista de classes;
- objetos importantes que ficaram sem anotação.

Uma anotação incorreta não é só um erro no arquivo: para o modelo, ela é a resposta correta que ele deve aprender.

---

## 2. `data.yaml`: o contrato entre dataset e modelo

O YOLO não adivinha onde seu dataset está nem o significado de `0`, `1` ou `2` nos arquivos de label. Essas informações ficam no `data.yaml`.

```yaml
path: /caminho/para/meu-dataset
train: images/train
val: images/val

names:
  0: capacete
  1: sem_capacete
```

Leia esse arquivo como um contrato:

| Campo | Função | Erro comum |
|---|---|---|
| `path` | Diretório raiz do dataset | apontar para a pasta errada |
| `train` | Imagens usadas para aprender | informar uma pasta que não existe |
| `val` | Imagens usadas para avaliar | reutilizar as imagens de treino |
| `names` | Tradução dos índices para nomes | mudar a ordem das classes |

Se um label começa com `1`, neste exemplo ele sempre significa `sem_capacete`. A ordem da lista de classes precisa ser exatamente a mesma que foi usada na anotação.

---

## 3. Começar de um modelo que já enxerga

Treinar do zero exigiria muito mais imagens e processamento. Nesta aula, usamos *transfer learning*: começamos com um YOLO pré-treinado e adaptamos seus pesos ao novo dataset.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

O sufixo `n` representa uma versão pequena do modelo. Ela é uma boa escolha para experimentar em aula porque treina mais rápido. Modelos maiores podem produzir resultados melhores, mas aumentam tempo de treino e uso de memória.

```text
pesos pré-treinados
        +
suas imagens anotadas
        ↓
pesos adaptados ao seu domínio
```

O modelo não ganha conhecimento mágico sobre uma nova classe. Ele usa padrões visuais já aprendidos, como contornos, texturas e partes de objetos, para aprender a localizar a classe que você anotou.

---

## 4. Executando o treino

O treinamento básico cabe em poucas linhas:

```python
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="capacete_v1"
)
```

| Parâmetro | O que controla | Pergunta prática |
|---|---|---|
| `data` | Arquivo de configuração | O caminho e as classes estão corretos? |
| `epochs` | Quantas vezes o modelo percorre o treino | Ele ainda está melhorando? |
| `imgsz` | Tamanho usado na entrada do modelo | Objetos pequenos precisam de mais detalhe? |
| `batch` | Quantas imagens são processadas por vez | Cabe na memória disponível? |
| `name` | Pasta que guardará o resultado | Consigo identificar este experimento depois? |

### Uma época não é uma imagem

Uma **época** termina quando o modelo passou por todas as imagens de `train` uma vez. Em cada passagem, ele compara suas previsões com os labels e ajusta os pesos para reduzir o erro.

Mais épocas não garantem um modelo melhor. Depois de certo ponto, o modelo pode melhorar apenas nas imagens de treino e piorar em imagens novas. Esse comportamento é chamado de **overfitting**.

!!! tip "Primeiro experimento"
    Antes de um treino longo, rode poucas épocas apenas para testar o pipeline. É a forma mais barata de descobrir caminhos quebrados, labels inválidos ou classes invertidas.

---

## 5. O que o YOLO avalia durante o treino?

Para cada previsão, o modelo precisa acertar três coisas ao mesmo tempo:

```text
classe correta + caixa bem posicionada + confiança adequada
```

Isso explica por que uma métrica única não basta para detecção. Uma previsão de `capacete` pode ter a classe certa, mas uma caixa deslocada. Ela ainda é um resultado ruim.

### IoU: a caixa está no lugar certo?

O **Intersection over Union** compara a caixa prevista com a caixa anotada:

$$
IoU = \frac{\text{área de sobreposição}}{\text{área de união}}
$$

- `IoU = 0`: as caixas não se encontram;
- `IoU = 1`: as caixas são idênticas;
- um limiar como `IoU >= 0.5` define o quanto a caixa precisa coincidir para contar como uma boa localização.

### Explore IoU, confiança e NMS

Arraste as caixas coloridas no simulador. A caixa tracejada é a anotação real (*ground truth*). O IoU mostrado é a sobreposição entre essa anotação e a previsão vermelha. Depois, altere os limiares para observar duas decisões diferentes: o filtro por confiança remove previsões fracas; o NMS remove caixas duplicadas para o mesmo objeto.

<div id="deteccao-widget" style="border: 1px solid #b7c6c2; padding: 16px; margin: 20px 0; background: #f8fbf9;" markdown="1">
<div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px;">
    <label>Confiança mínima <input data-confidence-threshold type="range" min="0" max="100" step="5" value="50"> <output data-confidence-value>50%</output></label>
    <label>IoU para NMS <input data-iou-threshold type="range" min="0" max="100" step="5" value="50"> <output data-iou-value>50%</output></label>
    <button type="button" data-reset-boxes>Reiniciar</button>
</div>
<canvas data-detection-canvas width="380" height="260" style="max-width: 100%; height: auto; border: 1px solid #b7c6c2; cursor: grab;" aria-label="Simulador de caixas delimitadoras"></canvas>
<div style="display: grid; gap: 5px; margin-top: 10px;">
    <span>IoU previsão × Ground Truth: <strong data-overlap-value></strong></span>
    <span data-nms-result></span>
</div>
</div>

### Precisão e recall: que tipo de erro ocorreu?

| Situação | O que significa |
|---|---|
| Verdadeiro positivo | detectou um objeto que existe, com classe e caixa aceitáveis |
| Falso positivo | desenhou uma caixa em algo que não deveria detectar |
| Falso negativo | deixou de detectar um objeto que existe |

A **precisão** responde: "entre as detecções exibidas, quantas estavam certas?". O **recall** responde: "entre os objetos que existiam, quantos foram encontrados?".

Ao elevar o limiar de confiança, você normalmente reduz falsos positivos, mas também pode aumentar falsos negativos. O melhor limiar depende da consequência de cada erro no problema real.

### Explore precisão e recall

O simulador representa oito objetos anotados como círculos tracejados. Quadrados vermelhos são detecções corretas; quadrados amarelos são falsos positivos. Mova o limiar de confiança e acompanhe o que muda nas métricas.

<div id="metricas-deteccao-widget" style="border: 1px solid #b7c6c2; padding: 16px; margin: 20px 0; background: #f8fbf9;" markdown="1">
<label>
    Limiar de confiança
    <input data-metrics-threshold type="range" min="0" max="100" step="5" value="50">
    <output data-metrics-threshold-value>50%</output>
</label>
<div style="display: flex; flex-wrap: wrap; gap: 18px; margin: 14px 0;">
    <strong>Precisão: <span data-metrics-precision></span></strong>
    <strong>Recall: <span data-metrics-recall></span></strong>
    <span data-metrics-counts></span>
</div>
<p data-metrics-calculation></p>
<div style="display: flex; flex-wrap: wrap; gap: 16px;">
    <canvas data-metrics-scene width="380" height="270" style="max-width: 100%; height: auto; border: 1px solid #b7c6c2;" aria-label="Detecções após o limiar de confiança"></canvas>
    <canvas data-metrics-curve width="380" height="240" style="max-width: 100%; height: auto; border: 1px solid #b7c6c2;" aria-label="Curva precisão recall"></canvas>
</div>
<p data-metrics-ap style="margin-bottom: 0;"></p>
</div>

### mAP: o resumo para comparar experimentos

O gráfico do simulador é construído assim:

```text
predições + anotações
    ↓
TP, FP e FN em cada limiar de confiança
    ↓
pontos de precisão e recall
    ↓
área sob a curva precisão-recall = AP de uma classe
    ↓
média dos APs das classes = mAP
```

O ponto vermelho marca a escolha atual de confiança. Ao mover o controle, ele muda de lugar porque TP, FP e FN mudam. A área azul, por sua vez, considera todos os limiares possíveis: ela é o **Average Precision (AP)** para uma classe.

Se o dataset tiver, por exemplo, `capacete` e `sem_capacete`, o YOLO calcula um AP para cada classe e tira a média. Esse resultado é o **mean Average Precision (mAP)**.

O YOLO costuma exibir, entre outras, estas medidas:

| Métrica | Como ler |
|---|---|
| `mAP50` | considera uma previsão correta a partir de $IoU = 0.50$ |
| `mAP50-95` | testa vários limiares de IoU, de 0.50 a 0.95; é mais rigorosa |

Em `mAP50`, uma previsão só precisa ter $IoU \geq 0.50$ para poder ser considerada correta. Em `mAP50-95`, o cálculo é repetido com vários critérios de IoU, por isso o valor costuma ser menor e mais exigente.

Use o mAP para comparar versões treinadas no mesmo dataset. Não trate um número isolado como aprovação automática: veja também exemplos visuais e as métricas por classe.

---

## 6. Lendo os artefatos gerados

Após o treino, a pasta `runs/detect/capacete_v1/` reúne os resultados. Os arquivos mais úteis são:

| Arquivo | Como usar |
|---|---|
| `weights/best.pt` | Melhor checkpoint segundo a validação |
| `weights/last.pt` | Estado da última época executada |
| `results.png` | Curvas de perda e métricas ao longo das épocas |
| `confusion_matrix.png` | Quais classes foram confundidas entre si |
| `val_batch*_pred.jpg` | Comparação visual das previsões com dados de validação |

Ao abrir `results.png`, procure tendências, não uma época específica:

```text
perda de treino cai e validação melhora      -> aprendizado consistente
perda de treino cai, validação piora         -> possível overfitting
ambas permanecem altas                        -> modelo ainda não aprendeu bem
```

Se uma classe está muito pior que as demais, investigue primeiro a quantidade e a qualidade das anotações daquela classe. Ajustar hiperparâmetros não corrige um dataset que não representa o problema.

---

## 7. Testando uma imagem nova

O teste final deve usar uma imagem que não pertence a `train` nem a `val`.

```python
from ultralytics import YOLO

model = YOLO("runs/detect/capacete_v1/weights/best.pt")

results = model.predict(
    source="teste/obra_nova.jpg",
    conf=0.40,
    save=True
)
```

Compare o resultado com a cena real. Faça perguntas concretas:

- detectou os objetos que importam?
- há caixas em regiões erradas?
- a classe está correta?
- objetos pequenos, parcialmente ocultos ou em outra iluminação ainda funcionam?

Esse teste revela a **generalização**: a capacidade de responder bem fora das imagens usadas para ajustar o modelo.

---

## 8. Quando o resultado não está bom

Antes de aumentar épocas ou trocar de modelo, diagnostique o tipo de falha.

| Sintoma | Hipótese inicial | Próxima ação |
|---|---|---|
| Não detecta quase nada | poucas imagens ou labels incorretos | revisar anotações e diversidade do dataset |
| Detecta objetos inexistentes | contexto visual insuficiente | incluir imagens negativas e fundos variados |
| Confunde duas classes | classes visualmente parecidas ou mal definidas | revisar a regra de anotação e coletar exemplos difíceis |
| Caixa deslocada | bounding boxes inconsistentes | inspecionar labels em imagens variadas |
| Funciona só em cenas parecidas com o treino | dataset pouco diverso | adicionar iluminação, ângulos e distâncias diferentes |

Um modelo melhor quase sempre começa com dados melhores. Treinamento é um ciclo: observar os erros, melhorar o dataset, treinar uma nova versão e comparar os resultados.

## Próximo passo

Use o notebook para executar um primeiro experimento com seu dataset. Registre o nome da execução, as métricas e alguns exemplos de acertos e erros. Esse registro permite comparar a próxima versão do modelo com evidências, não apenas com impressão visual.