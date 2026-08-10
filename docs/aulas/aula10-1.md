# Desafio — Transfer Learning com EuroSAT

Nesta atividade, você deverá aplicar os conceitos de **Transfer Learning** estudados na aula anterior em um novo problema de classificação.

Faça o download do notebook-base e siga as instruções para desenvolver sua solução.

[Baixar notebook-base](lab12/desafio_transfer_learning_eurosat.ipynb){ .md-button .md-button--primary }

---

## Cenário

Uma equipe está desenvolvendo uma solução para classificar automaticamente regiões observadas em imagens de satélite.

O objetivo é identificar o tipo de uso ou cobertura do solo presente em cada imagem, como áreas residenciais, florestas, rios, rodovias e plantações.

Sua tarefa é desenvolver um classificador utilizando uma arquitetura pré-treinada e comparar duas estratégias:

- **Feature Extraction**
- **Fine-Tuning**

Ao final, você deverá decidir qual abordagem apresentou o melhor resultado para o problema.

---

## Dataset

Utilize o **EuroSAT**, disponível no `torchvision`.

![EuroSAT](https://github.com/phelber/EuroSAT/blob/master/eurosat_overview_small.jpg?raw=true)

O dataset possui aproximadamente **27 mil imagens RGB**, distribuídas em **10 classes** de uso e cobertura do solo.

[Documentação do EuroSAT](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html){ .md-button }

- AnnualCrop (Culturas Anuais): Terras agrícolas com plantios de ciclo curto que precisam ser replantados todos os anos, como grãos e cereais.
- Forest (Floresta): Áreas densamente cobertas por árvores e vegetação florestal.
- HerbaceousVegetation (Vegetação Herbácea): Áreas de vegetação rasteira, gramíneas e plantas sem caule lenhoso.
- Highway (Rodovia): Estradas, autoestradas e principais vias de transporte pavimentadas.
- Industrial (Industrial): Prédios industriais, fábricas, armazéns e grandes instalações comerciais.
- Pasture (Pastagem): Áreas de pasto utilizadas para o manejo de gado e outros animais.
- PermanentCrop (Culturas Permanentes): Plantações de ciclo longo que não são destruídas após a colheita, como pomares, vinhedos e olivais.
- Residential (Residencial): Zonas urbanas ou rurais com foco em habitação, incluindo casas e prédios residenciais.
- River (Rio): Corpos de água lineares em movimento, como rios e riachos.
- SeaLake (Mar e Lago): Corpos de água parados ou abertos, cobrindo mares, oceanos, lagos e grandes lagoas.

---

## O desafio

Utilizando o notebook-base, desenvolva uma solução que:

1. carregue e explore o EuroSAT;
2. divida os dados em **70% treino, 15% validação e 15% teste**, usando `seed = 42`;
3. escolha uma arquitetura pré-treinada diferente da **MobileNetV3-Small**;
4. implemente **Feature Extraction**;
5. implemente **Fine-Tuning**;
6. compare os dois experimentos usando o conjunto de validação;
7. avalie o melhor modelo no conjunto de teste;
8. gere uma matriz de confusão e mostre exemplos de erros do modelo;
9. salve o melhor modelo e seus metadados.

!!! warning "Regra importante"
    O conjunto de **teste só deve ser utilizado no final**, depois que você já tiver escolhido o melhor experimento com base na validação.

---

## O que deve aparecer no notebook

Seu notebook deve mostrar, no mínimo:

- exemplos das imagens e classes do EuroSAT;
- arquitetura escolhida e justificativa;
- parâmetros usados no treinamento;
- curvas de loss e acurácia;
- comparação entre Feature Extraction e Fine-Tuning;
- resultado final no teste;
- matriz de confusão;
- pelo menos **8 classificações incorretas**;
- uma conclusão curta dizendo qual solução você escolheria.

---

## Entrega

Entregue o notebook **executado**, com código, gráficos, métricas e respostas visíveis.

O notebook deve executar do início ao fim no Google Colab sem depender de arquivos locais do computador.

!!! note
    O objetivo não é apenas obter a maior acurácia.  
    O mais importante é demonstrar que você sabe **aplicar, comparar e avaliar Transfer Learning** em um novo problema.
