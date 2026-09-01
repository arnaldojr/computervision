document.addEventListener("DOMContentLoaded", function () {
  const root = document.getElementById("deteccao-widget");
  if (!root) return;

  // =========================================================
  // ELEMENTOS DA INTERFACE
  // =========================================================
  const canvas = root.querySelector("[data-detection-canvas]");
  if (!canvas) return;

  const context = canvas.getContext("2d");

  const confidenceInput = root.querySelector("[data-confidence-threshold]");
  const iouInput = root.querySelector("[data-iou-threshold]");

  const confidenceValue = root.querySelector("[data-confidence-value]");
  const iouValue = root.querySelector("[data-iou-value]");
  const overlapValue = root.querySelector("[data-overlap-value]");
  const nmsValue = root.querySelector("[data-nms-result]");

  const resetButton = root.querySelector("[data-reset-boxes]");

  if (
    !confidenceInput ||
    !iouInput ||
    !confidenceValue ||
    !iouValue ||
    !overlapValue ||
    !nmsValue
  ) {
    return;
  }

  // =========================================================
  // GROUND TRUTH
  // =========================================================
  // Caixa anotada manualmente no conjunto de dados.
  // Ela NÃO participa do NMS.
  const groundTruth = {
    x: 118,
    y: 78,
    width: 150,
    height: 142,
    label: "pessoa",
  };

  // =========================================================
  // PREDIÇÕES DO MODELO
  // =========================================================
  const initialPredictions = [
    {
      id: 1,
      x: 129,
      y: 87,
      width: 142,
      height: 133,
      confidence: 0.92,
      label: "pessoa",
      color: "#e05d3f",
    },
    {
      id: 2,
      x: 105,
      y: 70,
      width: 154,
      height: 145,
      confidence: 0.74,
      label: "pessoa",
      color: "#f1a208",
    },
    {
      id: 3,
      x: 25,
      y: 148,
      width: 86,
      height: 75,
      confidence: 0.38,
      label: "mochila",
      color: "#576b7a",
    },
  ];

  // Cria uma cópia para permitir movimentação e reset.
  const predictions = initialPredictions.map((prediction) => ({
    ...prediction,
  }));

  let dragging = null;

  let dragOffset = {
    x: 0,
    y: 0,
  };

  // =========================================================
  // CÁLCULO DA INTERSEÇÃO
  // =========================================================
  function intersectionArea(first, second) {
    const left = Math.max(first.x, second.x);
    const top = Math.max(first.y, second.y);

    const right = Math.min(
      first.x + first.width,
      second.x + second.width
    );

    const bottom = Math.min(
      first.y + first.height,
      second.y + second.height
    );

    const width = Math.max(0, right - left);
    const height = Math.max(0, bottom - top);

    return width * height;
  }

  // =========================================================
  // IoU — INTERSECTION OVER UNION
  // =========================================================
  function calculateIoU(first, second) {
    const intersection = intersectionArea(first, second);

    const firstArea = first.width * first.height;
    const secondArea = second.width * second.height;

    const union =
      firstArea +
      secondArea -
      intersection;

    if (union <= 0) return 0;

    return intersection / union;
  }

  // =========================================================
  // FILTRO POR CONFIANÇA
  // =========================================================
  function filterByConfidence(items, threshold) {
    return items.filter(
      (prediction) =>
        prediction.confidence >= threshold
    );
  }

  // =========================================================
  // NON-MAXIMUM SUPPRESSION — NMS
  // =========================================================
  /*
   * Fluxo:
   *
   * 1. Ordena as caixas pela confiança.
   * 2. Mantém a caixa mais confiante.
   * 3. Compara com caixas da MESMA classe.
   * 4. Calcula o IoU entre as predições.
   * 5. Se IoU > threshold, elimina a menos confiante.
   * 6. Repete até processar todas as caixas.
   *
   * Ground Truth não participa deste processo.
   */
  function applyNms(items, threshold) {
    const sorted = [...items].sort(
      (a, b) => b.confidence - a.confidence
    );

    const kept = [];
    const suppressed = [];

    while (sorted.length > 0) {
      const current = sorted.shift();

      kept.push(current);

      for (let index = sorted.length - 1; index >= 0; index--) {
        const candidate = sorted[index];

        // NMS tradicional é aplicado entre caixas
        // que representam a mesma classe.
        if (candidate.label !== current.label) {
          continue;
        }

        const overlap = calculateIoU(
          current,
          candidate
        );

        if (overlap > threshold) {
          suppressed.push({
            prediction: candidate,
            suppressedBy: current,
            iou: overlap,
          });

          sorted.splice(index, 1);
        }
      }
    }

    return {
      kept,
      suppressed,
    };
  }

  // =========================================================
  // CENA DE FUNDO
  // =========================================================
  function drawScene() {
    context.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    // Fundo
    context.fillStyle = "#e9f0ec";
    context.fillRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    // Céu
    context.fillStyle = "#b9d8df";
    context.fillRect(
      0,
      0,
      canvas.width,
      95
    );

    // Chão
    context.fillStyle = "#7ba36b";
    context.fillRect(
      0,
      95,
      canvas.width,
      canvas.height - 95
    );

    // Corpo
    context.fillStyle = "#f7f1df";
    context.fillRect(
      167,
      86,
      48,
      130
    );

    // Cabeça
    context.beginPath();

    context.arc(
      191,
      57,
      23,
      0,
      Math.PI * 2
    );

    context.fillStyle = "#d68a61";
    context.fill();

    // Tronco
    context.fillStyle = "#264653";
    context.fillRect(
      172,
      81,
      38,
      84
    );

    // Pernas
    context.fillStyle = "#f1a208";

    context.fillRect(
      168,
      160,
      18,
      72
    );

    context.fillRect(
      197,
      160,
      18,
      72
    );

    // Mochila
    context.fillStyle = "#6c4e3b";

    context.fillRect(
      35,
      170,
      58,
      44
    );
  }

  // =========================================================
  // DESENHO DAS BOUNDING BOXES
  // =========================================================
  function drawBox(
    box,
    label,
    color,
    dashed = false
  ) {
    context.save();

    context.strokeStyle = color;
    context.lineWidth = 3;

    if (dashed) {
      context.setLineDash([7, 5]);
    }

    context.strokeRect(
      box.x,
      box.y,
      box.width,
      box.height
    );

    context.setLineDash([]);

    context.font =
      "bold 13px sans-serif";

    const textWidth =
      context.measureText(label).width + 12;

    const labelHeight = 21;

    // Evita que o rótulo fique fora do canvas.
    const labelY =
      box.y - labelHeight < 0
        ? box.y
        : box.y - labelHeight;

    context.fillStyle = color;

    context.fillRect(
      box.x,
      labelY,
      textWidth,
      labelHeight
    );

    context.fillStyle = "#ffffff";

    context.fillText(
      label,
      box.x + 6,
      labelY + 15
    );

    context.restore();
  }

  // =========================================================
  // MENSAGEM DIDÁTICA SOBRE NMS
  // =========================================================
  function updateNmsMessage(
    confidenceThreshold,
    nmsThreshold,
    confidenceFiltered,
    nmsResult
  ) {
    const primary = predictions[0];
    const secondary = predictions[1];

    const secondaryPassedConfidence =
      confidenceFiltered.some(
        (prediction) =>
          prediction.id === secondary.id
      );

    // -------------------------------------------------------
    // Caso 1 — caixa removida antes do NMS
    // -------------------------------------------------------
    if (!secondaryPassedConfidence) {
      nmsValue.textContent =
        `A caixa "pessoa ${Math.round(
          secondary.confidence * 100
        )}%" foi removida pelo limiar de confiança. ` +
        `Sua confiança é ${Math.round(
          secondary.confidence * 100
        )}% e o mínimo definido é ${Math.round(
          confidenceThreshold * 100
        )}%.`;

      return;
    }

    // -------------------------------------------------------
    // Verifica se a segunda caixa foi eliminada pelo NMS
    // -------------------------------------------------------
    const suppression =
      nmsResult.suppressed.find(
        (item) =>
          item.prediction.id === secondary.id
      );

    if (suppression) {
      const percentage =
        suppression.iou * 100;

      nmsValue.textContent =
        `A caixa "pessoa ${Math.round(
          secondary.confidence * 100
        )}%" foi removida pelo NMS. ` +
        `O IoU entre as duas detecções é ${percentage.toFixed(
          1
        )}%, acima do limiar de ${Math.round(
          nmsThreshold * 100
        )}%. ` +
        `Por isso, permanece a caixa mais confiante (${Math.round(
          suppression.suppressedBy.confidence * 100
        )}%).`;

      return;
    }

    // -------------------------------------------------------
    // Caso em que ambas permanecem
    // -------------------------------------------------------
    const primaryPassedConfidence =
      confidenceFiltered.some(
        (prediction) =>
          prediction.id === primary.id
      );

    if (
      primaryPassedConfidence &&
      secondaryPassedConfidence
    ) {
      const overlap =
        calculateIoU(
          primary,
          secondary
        );

      nmsValue.textContent =
        `As duas detecções de pessoa permanecem. ` +
        `O IoU entre elas é ${(overlap * 100).toFixed(
          1
        )}%, abaixo ou igual ao limiar de NMS de ${Math.round(
          nmsThreshold * 100
        )}%.`;

      return;
    }

    nmsValue.textContent =
      "Não há duas detecções da mesma classe disponíveis para o NMS comparar.";
  }

  // =========================================================
  // RENDERIZAÇÃO PRINCIPAL
  // =========================================================
  function draw() {
    const confidenceThreshold =
      Number(confidenceInput.value) / 100;

    const nmsThreshold =
      Number(iouInput.value) / 100;

    // -------------------------------------------------------
    // ETAPA 1
    // Filtro por confiança
    // -------------------------------------------------------
    const confidenceFiltered =
      filterByConfidence(
        predictions,
        confidenceThreshold
      );

    // -------------------------------------------------------
    // ETAPA 2
    // NMS
    // -------------------------------------------------------
    const nmsResult =
      applyNms(
        confidenceFiltered,
        nmsThreshold
      );

    // -------------------------------------------------------
    // IoU previsão × Ground Truth
    // -------------------------------------------------------
    // Para fins didáticos, usamos a primeira predição
    // da pessoa como exemplo de comparação com a anotação.
    const primary =
      predictions[0];

    const groundTruthIoU =
      calculateIoU(
        primary,
        groundTruth
      );

    // -------------------------------------------------------
    // Desenha a cena
    // -------------------------------------------------------
    drawScene();

    // Ground Truth
    drawBox(
      groundTruth,
      "Ground Truth: pessoa",
      "#167d83",
      true
    );

    // Somente caixas que sobreviveram
    // ao filtro de confiança + NMS.
    nmsResult.kept.forEach(
      (prediction) => {
        drawBox(
          prediction,
          `${prediction.label} ${Math.round(
            prediction.confidence * 100
          )}%`,
          prediction.color,
          false
        );
      }
    );

    // -------------------------------------------------------
    // Atualiza interface
    // -------------------------------------------------------
    confidenceValue.textContent =
      `${Math.round(
        confidenceThreshold * 100
      )}%`;

    iouValue.textContent =
      `${Math.round(
        nmsThreshold * 100
      )}%`;

    // IMPORTANTE:
    // Este é IoU da previsão com Ground Truth,
    // NÃO o IoU usado diretamente pelo NMS.
    overlapValue.textContent =
      `${(
        groundTruthIoU * 100
      ).toFixed(1)}%`;

    updateNmsMessage(
      confidenceThreshold,
      nmsThreshold,
      confidenceFiltered,
      nmsResult
    );
  }

  // =========================================================
  // POSIÇÃO DO PONTEIRO
  // =========================================================
  function pointerPosition(event) {
    const rect =
      canvas.getBoundingClientRect();

    return {
      x:
        (event.clientX - rect.left) *
        (canvas.width / rect.width),

      y:
        (event.clientY - rect.top) *
        (canvas.height / rect.height),
    };
  }

  // =========================================================
  // IDENTIFICA QUAL CAIXA FOI SELECIONADA
  // =========================================================
  function boxAt(position) {
    // Percorre de trás para frente para selecionar
    // primeiro a caixa desenhada "por cima".
    for (
      let index = predictions.length - 1;
      index >= 0;
      index--
    ) {
      const box = predictions[index];

      const inside =
        position.x >= box.x &&
        position.x <=
          box.x + box.width &&
        position.y >= box.y &&
        position.y <=
          box.y + box.height;

      if (inside) {
        return box;
      }
    }

    return null;
  }

  // =========================================================
  // DRAG
  // =========================================================
  canvas.addEventListener(
    "pointerdown",
    (event) => {
      const position =
        pointerPosition(event);

      dragging =
        boxAt(position);

      if (!dragging) return;

      dragOffset = {
        x:
          position.x -
          dragging.x,

        y:
          position.y -
          dragging.y,
      };

      canvas.setPointerCapture(
        event.pointerId
      );
    }
  );

  canvas.addEventListener(
    "pointermove",
    (event) => {
      if (!dragging) return;

      const position =
        pointerPosition(event);

      dragging.x = Math.max(
        0,
        Math.min(
          canvas.width -
            dragging.width,

          position.x -
            dragOffset.x
        )
      );

      dragging.y = Math.max(
        23,
        Math.min(
          canvas.height -
            dragging.height,

          position.y -
            dragOffset.y
        )
      );

      draw();
    }
  );

  function stopDragging(event) {
    if (
      dragging &&
      event &&
      canvas.hasPointerCapture?.(
        event.pointerId
      )
    ) {
      canvas.releasePointerCapture(
        event.pointerId
      );
    }

    dragging = null;
  }

  canvas.addEventListener(
    "pointerup",
    stopDragging
  );

  canvas.addEventListener(
    "pointercancel",
    stopDragging
  );

  // =========================================================
  // SLIDERS
  // =========================================================
  confidenceInput.addEventListener(
    "input",
    draw
  );

  iouInput.addEventListener(
    "input",
    draw
  );

  // =========================================================
  // RESET
  // =========================================================
  if (resetButton) {
    resetButton.addEventListener(
      "click",
      () => {
        initialPredictions.forEach(
          (original, index) => {
            Object.assign(
              predictions[index],
              original
            );
          }
        );

        confidenceInput.value = 50;
        iouInput.value = 50;

        dragging = null;

        draw();
      }
    );
  }

  // =========================================================
  // PRIMEIRA RENDERIZAÇÃO
  // =========================================================
  draw();
});