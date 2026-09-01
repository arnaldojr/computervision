document.addEventListener("DOMContentLoaded", function () {
  const root = document.getElementById("deteccao-widget");
  if (!root) return;

  const canvas = root.querySelector("[data-detection-canvas]");
  const context = canvas.getContext("2d");
  const confidenceInput = root.querySelector("[data-confidence-threshold]");
  const iouInput = root.querySelector("[data-iou-threshold]");
  const confidenceValue = root.querySelector("[data-confidence-value]");
  const iouValue = root.querySelector("[data-iou-value]");
  const overlapValue = root.querySelector("[data-overlap-value]");
  const nmsValue = root.querySelector("[data-nms-result]");
  const resetButton = root.querySelector("[data-reset-boxes]");

  const groundTruth = { x: 118, y: 78, width: 150, height: 142 };
  const predictions = [
    { x: 129, y: 87, width: 142, height: 133, confidence: 0.92, label: "pessoa", color: "#e05d3f" },
    { x: 105, y: 70, width: 154, height: 145, confidence: 0.74, label: "pessoa", color: "#f1a208" },
    { x: 25, y: 148, width: 86, height: 75, confidence: 0.38, label: "mochila", color: "#576b7a" },
  ];
  let dragging = null;
  let dragOffset = { x: 0, y: 0 };

  function intersectionArea(first, second) {
    const left = Math.max(first.x, second.x);
    const top = Math.max(first.y, second.y);
    const right = Math.min(first.x + first.width, second.x + second.width);
    const bottom = Math.min(first.y + first.height, second.y + second.height);
    return Math.max(0, right - left) * Math.max(0, bottom - top);
  }

  function iou(first, second) {
    const intersection = intersectionArea(first, second);
    const union = first.width * first.height + second.width * second.height - intersection;
    return union === 0 ? 0 : intersection / union;
  }

  function drawScene() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#e9f0ec";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#b9d8df";
    context.fillRect(0, 0, canvas.width, 95);
    context.fillStyle = "#7ba36b";
    context.fillRect(0, 95, canvas.width, canvas.height - 95);
    context.fillStyle = "#f7f1df";
    context.fillRect(167, 86, 48, 130);
    context.beginPath();
    context.arc(191, 57, 23, 0, Math.PI * 2);
    context.fillStyle = "#d68a61";
    context.fill();
    context.fillStyle = "#264653";
    context.fillRect(172, 81, 38, 84);
    context.fillStyle = "#f1a208";
    context.fillRect(168, 160, 18, 72);
    context.fillRect(197, 160, 18, 72);
    context.fillStyle = "#6c4e3b";
    context.fillRect(35, 170, 58, 44);
  }

  function drawBox(box, label, color, dashed) {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = 3;
    if (dashed) context.setLineDash([7, 5]);
    context.strokeRect(box.x, box.y, box.width, box.height);
    context.setLineDash([]);
    context.font = "bold 13px sans-serif";
    const text = label;
    const textWidth = context.measureText(text).width + 12;
    context.fillStyle = color;
    context.fillRect(box.x, box.y - 22, textWidth, 21);
    context.fillStyle = "#ffffff";
    context.fillText(text, box.x + 6, box.y - 7);
    context.restore();
  }

  function draw() {
    const confidenceThreshold = Number(confidenceInput.value) / 100;
    const iouThreshold = Number(iouInput.value) / 100;
    const primary = predictions[0];
    const secondary = predictions[1];
    const primaryIou = iou(primary, groundTruth);
    const duplicateIou = iou(primary, secondary);
    const visible = predictions.filter((prediction) => prediction.confidence >= confidenceThreshold);
    const keepSecondary = secondary.confidence < confidenceThreshold || duplicateIou <= iouThreshold;

    drawScene();
    drawBox(groundTruth, "anotacao real", "#167d83", true);
    visible.forEach((prediction, index) => {
      if (index === 1 && !keepSecondary) return;
      drawBox(prediction, `${prediction.label} ${(prediction.confidence * 100).toFixed(0)}%`, prediction.color, false);
    });

    confidenceValue.textContent = `${confidenceInput.value}%`;
    iouValue.textContent = `${iouInput.value}%`;
    overlapValue.textContent = `${(primaryIou * 100).toFixed(1)}%`;
    nmsValue.textContent = keepSecondary
      ? "A segunda caixa permanece: sobreposicao abaixo do limiar de NMS."
      : "A segunda caixa e removida: representa a mesma pessoa que a caixa mais confiante.";
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (canvas.width / rect.width),
      y: (event.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function boxAt(position) {
    return predictions.find((box) => (
      position.x >= box.x && position.x <= box.x + box.width &&
      position.y >= box.y && position.y <= box.y + box.height
    ));
  }

  canvas.addEventListener("pointerdown", (event) => {
    const position = pointerPosition(event);
    dragging = boxAt(position);
    if (!dragging) return;
    dragOffset = { x: position.x - dragging.x, y: position.y - dragging.y };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const position = pointerPosition(event);
    dragging.x = Math.max(0, Math.min(canvas.width - dragging.width, position.x - dragOffset.x));
    dragging.y = Math.max(23, Math.min(canvas.height - dragging.height, position.y - dragOffset.y));
    draw();
  });
  canvas.addEventListener("pointerup", () => { dragging = null; });
  confidenceInput.addEventListener("input", draw);
  iouInput.addEventListener("input", draw);
  resetButton.addEventListener("click", () => {
    predictions[0].x = 129; predictions[0].y = 87;
    predictions[1].x = 105; predictions[1].y = 70;
    predictions[2].x = 25; predictions[2].y = 148;
    confidenceInput.value = 50;
    iouInput.value = 50;
    draw();
  });

  draw();
});
