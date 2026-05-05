# Aula 8 — Introducao ao PyTorch e CNN na Pratica

Nesta aula voce vai dar o primeiro passo formal em **Deep Learning aplicado a Visao Computacional** usando **PyTorch**.

A ideia central e simples: sair de modelos totalmente conectados e entender por que as **Convolutional Neural Networks (CNNs)** sao mais adequadas para imagens.

Vamos manter o mesmo estilo de projeto que voce ja conhece no curso:

```text
Notebook de treino -> artifacts do modelo -> API FastAPI local
```

[Lab10 — Notebook CNN com PyTorch](lab10/mnist-api/01_treinando_cnn_pytorch.ipynb){ .md-button .md-button-primary }

[Lab10 — Demo API (Cookbook)](lab10/demo.md){ .md-button .md-button-primary }

## Objetivos de aprendizagem

Ao final da aula, voce deve ser capaz de:

1. Explicar o papel do **PyTorch** no ciclo de treinamento de modelos.
2. Entender como imagens viram **tensores** no formato esperado pela rede.
3. Diferenciar um modelo MLP de uma arquitetura **CNN**.
4. Implementar uma CNN basica com `Conv2d`, `ReLU` e `MaxPool2d`.
5. Executar um loop de treino com `loss`, `optimizer` e avaliacao.
6. Salvar `artifacts` para reutilizar o modelo em uma API local.

## Por que PyTorch?

PyTorch e uma biblioteca muito usada para construir e treinar redes neurais.

Ela e forte em tres pontos:

- **clareza de codigo**,
- **flexibilidade para experimentar**,
- **ecossistema amplo** para pesquisa e aplicacao.

!!! tip "Leitura pratica"

    Pense no PyTorch como o "motor" do modelo: ele cuida dos tensores, dos gradientes e da atualizacao dos pesos.

## Pipeline da aula

Nesta Aula 8, vamos seguir este fluxo:

```text
imagem -> tensor -> CNN -> loss -> backward -> update
```

No fechamento da aula:

```text
modelo treinado -> artifacts -> API local para inferencia
```

## Tensores: a base de tudo

No PyTorch, dados sao representados como tensores.

Para imagens, um formato comum e:

```text
[batch, canais, altura, largura]
```

Exemplo:

- imagem em escala de cinza: 1 canal
- imagem colorida: 3 canais

<quiz>
Qual formato representa corretamente um batch de 64 imagens RGB de 32x32?
- [ ] `[64, 32, 32]`
- [x] `[64, 3, 32, 32]`
- [ ] `[3, 64, 32, 32]`
- [ ] `[32, 32, 64, 3]`

</quiz>

## Do MLP para CNN: o que muda?

No MLP, normalmente a imagem e "achatada" em um vetor.

Em CNN, preservamos a estrutura espacial da imagem e aplicamos filtros locais.

Isso traz vantagens:

- melhor captura de padroes visuais,
- menor numero de parametros em relacao a MLP grande,
- melhor desempenho em tarefas de visao.

## Blocos fundamentais da CNN

### Convolucao (`Conv2d`)

Aplica filtros sobre a imagem para detectar padroes como bordas, texturas e formas.

### Ativacao (`ReLU`)

Introduz nao linearidade para aumentar a capacidade de representacao da rede.

### Pooling (`MaxPool2d`)

Reduz a dimensao espacial e ajuda a tornar o modelo mais robusto.

### Classificador final

Depois dos blocos convolucionais, usamos camadas lineares para gerar as classes finais.

<quiz>
Qual componente reduz a resolucao espacial dos mapas de ativacao?
- [ ] `ReLU`
- [ ] `Flatten`
- [x] `MaxPool2d`
- [ ] `CrossEntropyLoss`

</quiz>

## Estrutura de treino no PyTorch

O ciclo principal de treino segue este padrao:

1. `model.train()`
2. `optimizer.zero_grad()`
3. `logits = model(x)`
4. `loss = criterion(logits, y)`
5. `loss.backward()`
6. `optimizer.step()`

!!! warning "Erro classico"

    Esquecer `optimizer.zero_grad()` faz os gradientes acumularem indevidamente e prejudica o treino.

## Avaliacao e interpretacao

Na etapa de avaliacao:

- usar `model.eval()`,
- usar `torch.no_grad()`,
- observar metricas e exemplos de erro.

Nao basta olhar so acuracia: vale analisar onde o modelo confunde classes parecidas.

## Fechamento com API local (Cookbook)

A aula termina com a conexao entre treino e aplicacao.

Vamos reutilizar o cookbook do curso para inferencia local:

1. gerar os `artifacts` no notebook,
2. carregar os artifacts no projeto da API,
3. testar predicao localmente.

!!! info "Mensagem principal da trilha"

    O aluno nao aprende apenas a treinar modelo.
    Ele aprende a transformar o modelo em um componente de sistema.

