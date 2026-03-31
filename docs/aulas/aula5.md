# Aula 5 — Template Matching: encontrando padrões em imagens

Nesta aula você vai avançar para uma técnica clássica de **busca por padrão visual**: o **template matching**.

A ideia central é simples e poderosa: dado um pequeno recorte de referência (**template**), queremos localizar onde esse padrão aparece dentro de uma imagem maior. Esse tipo de abordagem é útil em cenários controlados, interfaces, jogos, inspeção visual simples e problemas em que o objeto procurado mantém aparência, escala e orientação relativamente estáveis.

[Lab06 — Template Matching (Notebook)](lab06/arquivolab6.zip){ .md-button .md-button-primary }

[Mini Projeto - Xadrez](lab06/arquivoproj.zip){ .md-button .md-button-primary }


## Objetivos de aprendizagem

Ao final da aula, você deve ser capaz de:

1. Explicar o que é **template matching** e em quais situações ele funciona bem.
2. Diferenciar **template** e **imagem de análise**.
3. Aplicar `cv2.matchTemplate()` para localizar um padrão em uma imagem.
4. Interpretar o **mapa de correspondência** gerado pelo algoritmo.
5. Usar `cv2.minMaxLoc()` para extrair a melhor posição de correspondência.
6. Desenhar corretamente uma **bounding box** ao redor do objeto detectado.
7. Comparar diferentes métodos de template matching do OpenCV.
8. Detectar **múltiplas ocorrências** de um mesmo padrão usando limiarização do mapa de resposta.
9. Utilizar `np.where()` e `zip()` para percorrer todas as posições candidatas.
10. Analisar criticamente as **limitações** da técnica.

## O que é Template Matching?

O **template matching** é uma técnica de visão computacional usada para procurar uma pequena imagem de referência dentro de uma imagem maior.

Temos dois elementos principais:

- **Template:** o recorte do padrão que queremos encontrar
- **Imagem de análise:** a imagem maior na qual vamos procurar esse padrão

A intuição é parecida com deslizar uma pequena janela sobre a imagem e, em cada posição, perguntar:

> “O quanto essa região se parece com o template?”

A resposta para todas as posições forma um **mapa de correspondência**.

<quiz>
No contexto de template matching, o que a função principal do OpenCV retorna?
- [ ] A lista de contornos do objeto
- [ ] A imagem binarizada do objeto
- [x] Um mapa de correspondência com o grau de similaridade em cada posição
- [ ] Um conjunto de bounding boxes já prontas

</quiz>

### O template precisa representar o que você quer encontrar

O template é um pequeno recorte daquilo que você deseja localizar. Quanto mais coerente ele for com o objeto presente na imagem, maior a chance de a correspondência funcionar.

!!! warning "Erro clássico"

    Escolher um template mal recortado, com excesso de fundo, baixa nitidez ou proporções inconsistentes costuma degradar bastante o resultado.

### `matchTemplate()` não devolve a caixa pronta

A função não devolve diretamente o retângulo do objeto. Ela devolve um **mapa de resposta**: uma matriz onde cada posição representa o quanto o template combina com aquela região da imagem.

!!! tip "Como depurar rápido"

    Sempre mostre:

    1) a imagem original  
    2) o template  
    3) o mapa de correspondência  
    4) a imagem final com a caixa desenhada

### Escala de cinza simplifica a comparação

No notebook, a comparação principal é feita em **tons de cinza**. Isso reduz a complexidade e ajuda a focar no padrão estrutural do objeto.

!!! info "Observação prática"

    Converter para tons de cinza não torna a técnica invariável à iluminação. Apenas simplifica a entrada para a comparação.

## Como o algoritmo funciona?

O pipeline conceitual desta aula é:

1. Carregar a imagem maior.
2. Carregar o template.
3. Converter as imagens para um formato adequado (no notebook, **grayscale**).
4. Aplicar `cv2.matchTemplate(imagem, template, método)`.
5. Interpretar o mapa de resposta.
6. Escolher a melhor correspondência ou várias correspondências.
7. Desenhar as caixas sobre a imagem original.


## Parte A — Encontrando uma ocorrência única

No notebook, a primeira demonstração busca localizar um **GOOMBA** em uma cena do jogo.

A operação central é:

```python
res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
```

Nesse caso, usamos o método `TM_SQDIFF`, baseado em diferença quadrática.

!!! info "Interpretação importante"

    No método `TM_SQDIFF`, valores **menores** representam melhores correspondências.

O mapa de resposta é uma imagem numérica. Cada posição indica o quanto o template se ajusta àquela região da cena.
Visualmente, a melhor correspondência tende a aparecer como uma região mais escura no mapa.

<quiz>
Usando `cv2.TM_SQDIFF`, qual posição deve ser escolhida como melhor correspondência?
- [x] A posição do valor mínimo
- [ ] A posição do valor máximo
- [ ] A média entre mínimo e máximo
- [ ] A posição central do mapa

</quiz>

### a melhor posição com `minMaxLoc()`

Para transformar o mapa em coordenadas utilizáveis, usamos:

```python
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
```

No método `TM_SQDIFF`:

- `min_val` é o melhor valor encontrado
- `min_loc` é a posição do canto superior esquerdo da melhor correspondência

Essa posição será o ponto de início da caixa que vamos desenhar.

Depois de localizar a melhor correspondência, o próximo passo é desenhar uma caixa sobre a imagem original.

Sabemos que:

- o ponto inicial é `min_loc`
- o template tem altura e largura conhecidas

Logo:

- `top_left = min_loc`
- `bottom_right = (min_loc[0] + largura, min_loc[1] + altura)`

E então usamos `cv2.rectangle()`.

```python
cv2.rectangle(img_result, top_left, bottom_right, (0, 255, 255), 4)
```

!!! warning "Erro clássico"

    É comum confundir a ordem retornada por `template.shape`.
    Em imagens em tons de cinza, o retorno é `(altura, largura)`.

## Comparando diferentes métodos

O OpenCV fornece vários métodos de template matching. Todos têm a mesma assinatura de uso, mas não a mesma interpretação.

- `cv2.TM_SQDIFF` 
- `cv2.TM_SQDIFF_NORMED`
- `cv2.TM_CCORR`
- `cv2.TM_CCORR_NORMED`
- `cv2.TM_CCOEFF`
- `cv2.TM_CCOEFF_NORMED`

Há um ponto conceitual central aqui:

- Para **`TM_SQDIFF`** e **`TM_SQDIFF_NORMED`**:  
  **quanto menor, melhor**

- Para **`TM_CCORR`**, **`TM_CCORR_NORMED`**, **`TM_CCOEFF`** e **`TM_CCOEFF_NORMED`**:  
  **quanto maior, melhor**

!!! tip "Regra mental útil"

    Pergunte sempre: este método mede “erro” ou “similaridade”?  
    - Se mede erro, o melhor valor é o menor.  
    - Se mede similaridade/correlação, o melhor valor é o maior.

Mas por que comparar métodos? Porque a qualidade da resposta depende de fatores como:

- contraste entre template e fundo,
- iluminação,
- textura do cenário,
- presença de padrões parecidos na imagem,
- necessidade ou não de normalização.

<quiz>
Ao usar `TM_CCOEFF_NORMED`, qual é a regra correta para escolher a melhor correspondência?
- [ ] Escolher o menor valor do mapa
- [x] Escolher o maior valor do mapa
- [ ] Escolher sempre o valor mais próximo de 0
- [ ] Escolher o primeiro valor acima da média

</quiz>

## Parte A — Detectando múltiplas ocorrências

Até aqui, usamos `minMaxLoc()` para encontrar **apenas uma** correspondência: a melhor.

Mas há cenários em que queremos encontrar **todas as ocorrências** do mesmo padrão na imagem. No notebook, isso é explorado com as **caixas de interrogação** do cenário do Mario.

Para múltiplas correspondências, o fluxo passa a ser:

1. gerar o mapa de resposta;
2. definir um **threshold**;
3. selecionar todas as posições que satisfazem o critério;
4. desenhar uma caixa para cada posição válida.

### O threshold

No notebook, com `TM_SQDIFF_NORMED`, o critério é do tipo:

```python
loc = np.where(res <= threshold)
```

Como o método mede erro normalizado:

- valores menores → melhores correspondências
- threshold baixo → seleção mais rígida
- threshold alto → seleção mais permissiva

!!! warning "Erro clássico"

    Um threshold permissivo demais gera muitos falsos positivos.  
    Um threshold rígido demais pode perder ocorrências válidas.

### Por que aparece `loc[::-1]`?

Porque o `np.where()` produz os índices na ordem:

- linhas (`y`)
- colunas (`x`)

Mas funções geométricas como `cv2.rectangle()` trabalham naturalmente com coordenadas na forma:

- `(x, y)`

Assim, inverter a ordem ajuda a percorrer os pontos no formato esperado para desenho.

!!! tip "Leitura prática"

    Pense assim:

    - `np.where()` → devolve “onde está” na lógica matricial  
    - `cv2.rectangle()` → precisa de ponto no sistema geométrico `(x, y)`


## Limitações do Template Matching

O template matching é útil e intuitivo, mas possui limitações importantes.

### Escala
Se o objeto aparece maior ou menor que o template, a correspondência tende a piorar bastante.

### Rotação
Se o objeto está rotacionado em relação ao template, a semelhança medida pode cair significativamente.

### Iluminação e contraste
Mudanças de brilho, sombras e contraste podem alterar a resposta, especialmente em métodos mais sensíveis à intensidade dos pixels.

### Oclusão parcial
Se parte do objeto estiver escondida, o padrão completo deixa de encaixar bem.

### Padrões repetitivos
Em imagens com muitas texturas semelhantes, o algoritmo pode encontrar vários falsos positivos.

!!! info

    Template matching funciona melhor em cenários controlados, com baixa variação de escala, rotação e iluminação.

<quiz>
Qual das alternativas descreve melhor uma limitação típica do template matching?
- [ ] Ele não funciona em imagens coloridas
- [ ] Ele só funciona com contornos fechados
- [x] Ele é sensível a variações de escala, rotação e iluminação
- [ ] Ele exige redes neurais convolucionais

</quiz>
