# Aula 6 — Features e Descritores

Nesta aula você vai avançar para uma abordagem mais robusta de visão computacional: o uso de **features**, **keypoints** e **descritores** para localizar objetos em imagens. Alem de conhecer a tecnica de transformação de geometrica por meio da matriz de **Homografia**. 

A ideia central é `simples`: em vez de comparar a imagem inteira com um recorte fixo, vamos detectar **pontos de interesse** e descrever matematicamente a vizinhança desses pontos. Isso permite realizar correspondências mais robustas, inclusive quando o objeto aparece com **rotação**, **mudança de escala** e pequenas variações de cena.


[Lab07 — Feature (Notebook)](lab07/arquivolab7.zip){ .md-button .md-button-primary }

[Lab08 — Perspectiva (Notebook)](lab08/arquivolab8.zip){ .md-button .md-button-primary }

## Objetivos de aprendizagem

Ao final da aula, você deve ser capaz de:

1. Explicar o que são **features** em visão computacional.
2. Diferenciar **keypoints** e **descritores**.
3. Detectar keypoints com o algoritmo **ORB** e **SIFT**.
4. Realizar **matching** entre descritores.
5. Conhecer a tecnica de **stitching panorâmico**.
6. Conhecer a tecnica de transformação geométrica **Homografia**

## Por que precisamos de features?

Na aula anterior, vimos que o **template matching** pode funcionar bem em cenários controlados. No entanto, essa técnica apresenta limitações importantes quando o objeto:

- muda de escala,
- aparece rotacionado,
- sofre variações de iluminação,
- está parcialmente ocluído,
- ou surge em uma cena mais complexa.

É justamente nesse tipo de situação que abordagens baseadas em **features** se tornam mais interessantes.

## O que são Features?

Em visão computacional, **features** são características detectáveis em regiões específicas da imagem e que ajudam a representar visualmente um objeto.

Na prática, estamos procurando estruturas que sejam suficientemente informativas para serem reconhecidas novamente em outra imagem.

Essas estruturas podem aparecer como:

- **cantos**,
- **junções**,
- **bordas marcantes**,
- **blobs**,
- **regiões texturizadas**.

### O que esperamos de uma boa feature?

Uma boa feature deve ser, na medida do possível:

- **repetível**, aparecendo novamente mesmo com pequenas mudanças na imagem;
- **distintiva**, para não ser confundida facilmente com outras;
- **local**, ocupando uma pequena vizinhança;
- **precisa**, para que sua posição seja bem determinada;
- **eficiente**, para permitir processamento viável;
- **numerosa**, em quantidade suficiente para representar o objeto.

!!! tip "Leitura prática"

    Pense assim: uma boa feature é um ponto que “vale a pena guardar” porque pode ser encontrado novamente em outra imagem.

<quiz>
Qual é uma vantagem importante das técnicas baseadas em features em relação ao template matching?
- [ ] Elas sempre dispensam qualquer etapa de filtragem
- [ ] Elas funcionam apenas em imagens binárias
- [x] Elas tendem a ser mais robustas a rotação e mudança de escala
- [ ] Elas devolvem automaticamente a segmentação perfeita do objeto

</quiz>

## Keypoints e Descritores

### Keypoints (Pontos de Interesse)

Keypoints são pontos específicos em uma imagem que possuem propriedades distintivas, como cantos, bordas ou regiões com alta variação de intensidade. São invariantes a transformações como rotação, escala e iluminação.

### Descritores

Descritores são vetores numéricos que codificam informações sobre a vizinhança de um keypoint, permitindo comparação entre diferentes keypoints.
<quiz>
O que descreve melhor um descritor?
- [ ] Um filtro que remove ruído da imagem
- [ ] Um retângulo desenhado sobre o objeto
- [x] Um vetor numérico que codifica a vizinhança de um keypoint
- [ ] Um método de binarização automática

</quiz>

## Detectando Features com ORB

O ORB é bastante popular porque oferece uma solução eficiente e prática para detectar e descrever features.

!!! info "Observação importante"

    ORB não é a técnica mais robusta em todos os cenários, mas é uma boa para aprender o pipeline completo de detecção, descrição e matching.

O `detectAndCompute()`, faz duas tarefas de uma vez:

1. detecta os keypoints;
2. calcula os descritores desses pontos.

Esse fluxo é importante porque, em feature matching, não basta encontrar pontos interessantes. É preciso também **descrevê-los** para poder compará-los.

<quiz>
Quando usamos `detectAndCompute()`, o que esperamos obter?
- [ ] Apenas a imagem com os contornos desenhados
- [x] Os keypoints e os descritores
- [ ] Apenas os descritores, sem keypoints
- [ ] Um mapa de calor da imagem

</quiz>

## Visualizando os keypoints

Uma etapa muito importante do processo é **visualizar** os keypoints detectados.

Isso ajuda a responder perguntas como:

- Onde o algoritmo está focando?
- Há keypoints suficientes?
- Eles estão concentrados em regiões úteis?
- O objeto tem textura suficiente para ser reconhecido?

!!! warning "Erro clássico"

    Detectar keypoints não significa, por si só, que o objeto será reconhecido corretamente.  
    O ponto central é a qualidade das features e a qualidade das correspondências.

## Matching de Features

Depois de detectar e descrever os keypoints, tanto na imagem de referência como na imagem do espaço de busca,precisamos descobrir quais descritores de uma imagem se parecem com os descritores da outra.

Essa etapa é chamada de **matching**.

### Brute-Force Matcher

A ideia do algoritmo **BFMatcher**, Brute-Force Matcher é simples:

- ele compara descritores da imagem 1 com descritores da imagem 2;
- mede a distância entre eles;
- e identifica quais pares são mais semelhantes.

Para descritores binários, como os produzidos pelo ORB, usa-se tipicamente a norma **Hamming** que é uma métrica usada para comparar duas sequências de bits e contar quantas posições correspondentes são diferentes. 

Exemplo Numérico

Imagine que temos dois descritores binários (vetores) de 8 bits:

- Vetor A: 1 0 1 1 1 0 0 1
- Vetor B: 1 0 0 1 0 0 0 1

Para calcular a distância de Hamming, comparamos bit a bit:

- A[0] vs B[0]: 1, 1 (Igual)
- A[1] vs B[1]: 0, 0 (Igual)
- A[2] vs B[2]: 1, 0 (Diferente) -> +1
- A[3] vs B[3]: 1, 1 (Igual)
- A[4] vs B[4]: 1, 0 (Diferente) -> +1
- A[5] vs B[5]: 0, 0 (Igual)
- A[6] vs B[6]: 0, 0 (Igual)
- A[7] vs B[7]: 1, 1 (Igual) 

Distância de Hamming = 2 (os vetores diferem em duas posições).

!!! info "Regra prática"

    ORB costuma ser associado a `cv2.NORM_HAMMING`.

<quiz>
Ao trabalhar com descritores do ORB, qual tipo de medida é geralmente usada no BFMatcher?
- [ ] `cv2.NORM_L2`
- [ ] Correlação cruzada normalizada
- [x] `cv2.NORM_HAMMING`
- [ ] Distância Manhattan em nível de pixel

</quiz>

## Filtrando correspondências

Nem todo match encontrado é bom. Em geral, haverá também correspondências ruins ou ambíguas.

Por isso, uma etapa essencial é **filtrar**.

No contexto do `DMatch`:

- **distância menor** → correspondência melhor
- **distância maior** → correspondência pior

Assim, uma estratégia comum é:

1. ordenar os matches por distância;
2. selecionar apenas os melhores;
3. visualizar o resultado filtrado.

<quiz>
Ao ordenar matches pela propriedade `distance`, o que estamos priorizando?
- [ ] Os matches com maior área
- [ ] Os matches com maior intensidade média
- [x] Os matches mais similares
- [ ] Os matches mais próximos do centro da imagem

</quiz>

## Um ponto importante sobre custo computacional

O BFMatcher é simples e didático, mas pode ser caro computacionalmente.

Isso acontece porque ele compara muitos descritores entre si, o que cresce rapidamente conforme o número de pontos aumenta.

Em problemas pequenos isso é aceitável. Em cenários maiores, outras abordagens podem ser mais adequadas.

## Implementando SIFT

Depois de entender o pipeline com ORB, vamos repetir a lógica com **SIFT**. Que é uma técnica muito conhecida por sua robustez na detecção e descrição de features.

Em comparação com ORB, ele costuma ser:

- mais robusto,
- mais pesado computacionalmente,
- e associado a descritores de natureza diferente.

### O que muda com SIFT?

Algumas diferenças importantes:

- o detector passa a ser `cv2.SIFT_create()`;
- o matching costuma usar `cv2.NORM_L2`;
- e uma estratégia comum é usar `knnMatch()` em vez de um match simples direto.

<quiz>
Ao trocar ORB por SIFT, qual alteração costuma fazer sentido no matcher?
- [ ] Trocar para `cv2.NORM_HAMMING`
- [x] Trocar para `cv2.NORM_L2`
- [ ] Remover a etapa de descritores
- [ ] Usar apenas `matchTemplate()`

</quiz>

## knnMatch e o teste de razão de Lowe

Quando usamos `knnMatch()`, em vez de pedir um único melhor match, pedimos os **k vizinhos mais próximos** de cada descritor.

Isso permite aplicar um critério mais confiável para filtragem: o **teste de razão de Lowe**.

A lógica é:

- pegamos o melhor e o segundo melhor candidato;
- comparamos suas distâncias;
- se o melhor for significativamente melhor do que o segundo, o match tende a ser confiável.

Essa ideia ajuda a reduzir ambiguidades.

!!! tip

    Não basta um candidato ser “bom”.  
    Em muitos casos, ele também precisa ser claramente melhor do que o segundo melhor candidato.

<quiz>
Qual é a principal ideia do teste de razão de Lowe?
- [ ] Comparar a área do objeto com a área da cena
- [ ] Verificar se o número de keypoints é par
- [x] Aceitar matches em que o melhor candidato é bem melhor que o segundo melhor
- [ ] Escolher sempre os matches com maior distância

</quiz>


## Aplicação prática: panoramas

Uma aplicação clássica de feature matching é o **stitching panorâmico**.

A ideia geral é:

1. detectar features em imagens sobrepostas;
2. encontrar correspondências entre elas;
3. alinhar geometricamente as imagens;
4. combinar os resultados em uma imagem maior.

O `cv2.Stitcher`, encapsula boa parte desse processo.

Features não servem apenas para reconhecer objetos. Elas também são fundamentais para **alinhamento entre imagens**.

<quiz>
No contexto de panoramas, para que o feature matching é útil?
- [ ] Para substituir completamente a câmera
- [ ] Para calcular apenas histogramas de cor
- [x] Para alinhar imagens com regiões em comum
- [ ] Para binarizar automaticamente a cena

</quiz>


<quiz>
Qual afirmação resume melhor a contribuição desta aula?
- [ ] Features substituem completamente todos os outros métodos de visão computacional
- [ ] Descritores são úteis apenas para panoramas
- [x] Keypoints e descritores permitem correspondências mais robustas entre imagens
- [ ] O objetivo principal é apenas desenhar círculos em cantos da imagem

</quiz>

## transformação geométrica Homografia

A ideia é apenas entender transformação geométrica **para que serve** dentro do exemplo prático.

Quando encontramos **matches** entre duas imagens, não basta saber apenas que existem pontos parecidos. Em muitos casos, queremos descobrir se esses pontos realmente representam **o mesmo objeto ou a mesma região** vista de outra forma.

É nesse momento que entra transformação geométrica:

- **`findHomography()`** → descobre a transformação
- **`perspectiveTransform()`** → aplica essa transformação em pontos

### `cv2.findHomography()`

Essa função é usada para calcular uma transformação geométrica entre dois conjuntos de pontos correspondentes.

Na prática, ela ajuda a responder algo como:

> “Dado esse conjunto de pontos da imagem A e seus correspondentes na imagem B, qual transformação melhor relaciona uma imagem com a outra?”

No nosso exemplo, isso é útil para estimar onde o objeto da imagem de referência aparece na cena analisada.

### `cv2.perspectiveTransform()`

Depois que a homografia é calculada, essa função pode ser usada para **transformar pontos** de uma imagem para a outra.

Os **quatro cantos da imagem de referência** e projetado na cena. Assim, conseguimos desenhar na imagem final a região onde o objeto foi encontrado.

<quiz>
No contexto desta aula, qual é o papel mais prático de `findHomography()` e `perspectiveTransform()`?
- [ ] Substituir o detector de keypoints
- [ ] Melhorar automaticamente a nitidez da imagem
- [x] Estimar a transformação entre as imagens e projetar a posição do objeto na cena
- [ ] Converter a imagem para tons de cinza

</quiz>