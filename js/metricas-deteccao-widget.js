document.addEventListener("DOMContentLoaded", function () {
  const root = document.getElementById("metricas-deteccao-widget");
  if (!root) return;

  const thresholdInput = root.querySelector("[data-metrics-threshold]");
  const thresholdValue = root.querySelector("[data-metrics-threshold-value]");
  const precisionValue = root.querySelector("[data-metrics-precision]");
  const recallValue = root.querySelector("[data-metrics-recall]");
  const apValue = root.querySelector("[data-metrics-ap]");
  const countsValue = root.querySelector("[data-metrics-counts]");
    const calculationValue = root.querySelector("[data-metrics-calculation]");
  const sceneCanvas = root.querySelector("[data-metrics-scene]");
  const curveCanvas = root.querySelector("[data-metrics-curve]");

  if (!thresholdInput || !thresholdValue || !precisionValue || !recallValue ||
      !apValue || !countsValue || !calculationValue || !sceneCanvas || !curveCanvas) return;

  const scene = sceneCanvas.getContext("2d");
  const curve = curveCanvas.getContext("2d");

  const groundTruth = [
    { x: 46, y: 66 }, { x: 110, y: 120 }, { x: 177, y: 55 }, { x: 241, y: 123 },
    { x: 308, y: 70 }, { x: 71, y: 202 }, { x: 163, y: 210 }, { x: 269, y: 205 },
  ];

  const predictions = [
    { x: 46, y: 66, confidence: 0.97, match: true },
    { x: 110, y: 120, confidence: 0.91, match: true },
    { x: 177, y: 55, confidence: 0.84, match: true },
    { x: 241, y: 123, confidence: 0.72, match: true },
    { x: 308, y: 70, confidence: 0.63, match: true },
    { x: 71, y: 202, confidence: 0.51, match: true },
    { x: 163, y: 210, confidence: 0.37, match: true },
    { x: 269, y: 205, confidence: 0.24, match: true },
    { x: 330, y: 145, confidence: 0.88, match: false },
    { x: 25, y: 158, confidence: 0.58, match: false },
    { x: 210, y: 175, confidence: 0.42, match: false },
  ];

  function metricsFor(threshold) {
    const visible = predictions.filter((item) => item.confidence >= threshold);
    const truePositives = visible.filter((item) => item.match).length;
    const falsePositives = visible.length - truePositives;
    const falseNegatives = groundTruth.length - truePositives;
    const precision = visible.length ? truePositives / visible.length : 0;
    const recall = truePositives / groundTruth.length;

    return { visible, truePositives, falsePositives, falseNegatives, precision, recall };
  }

  function averagePrecision() {
    const ordered = [...predictions].sort((first, second) => second.confidence - first.confidence);
    let truePositives = 0;
    let previousRecall = 0;
    let area = 0;

    ordered.forEach((prediction, index) => {
      if (prediction.match) truePositives += 1;
      const precision = truePositives / (index + 1);
      const recall = truePositives / groundTruth.length;
      area += precision * (recall - previousRecall);
      previousRecall = recall;
    });

    return area;
  }

  function drawScene(data) {
    scene.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
    scene.fillStyle = "#eff4ef";
    scene.fillRect(0, 0, sceneCanvas.width, sceneCanvas.height);

    scene.font = "12px sans-serif";
    scene.fillStyle = "#31505b";
    scene.fillText("Objetos anotados", 12, 20);
    groundTruth.forEach((item) => {
      scene.beginPath();
      scene.arc(item.x, item.y, 13, 0, Math.PI * 2);
      scene.strokeStyle = "#167d83";
      scene.lineWidth = 2;
      scene.setLineDash([4, 3]);
      scene.stroke();
      scene.setLineDash([]);
    });

    data.visible.forEach((item) => {
      scene.fillStyle = item.match ? "#e05d3f" : "#f1a208";
      scene.fillRect(item.x - 10, item.y - 10, 20, 20);
      scene.fillStyle = "#ffffff";
      scene.font = "bold 10px sans-serif";
      scene.fillText(`${Math.round(item.confidence * 100)}`, item.x - 7, item.y + 4);
    });

    scene.fillStyle = "#31505b";
    scene.font = "12px sans-serif";
    scene.fillText("Quadrado vermelho: TP    Quadrado amarelo: FP", 12, 254);
  }

  function drawCurve(selected) {
    const points = [];
    for (let step = 0; step <= 20; step++) {
      const threshold = step / 20;
      const data = metricsFor(threshold);
      points.push({ recall: data.recall, precision: data.precision });
    }

    curve.clearRect(0, 0, curveCanvas.width, curveCanvas.height);
    curve.fillStyle = "#ffffff";
    curve.fillRect(0, 0, curveCanvas.width, curveCanvas.height);

    const left = 42;
    const top = 16;
    const width = 290;
    const height = 190;

    curve.strokeStyle = "#9aaca8";
    curve.lineWidth = 1;
    curve.beginPath();
    curve.moveTo(left, top);
    curve.lineTo(left, top + height);
    curve.lineTo(left + width, top + height);
    curve.stroke();

    curve.fillStyle = "#31505b";
    curve.font = "12px sans-serif";
    curve.fillText("Precisão", 5, 15);
    curve.fillText("Recall", 285, 228);
    curve.fillText("1", 25, 23);
    curve.fillText("1", 338, 210);

    const ordered = [...points].sort((first, second) => first.recall - second.recall);
    curve.beginPath();
    ordered.forEach((point, index) => {
      const x = left + point.recall * width;
      const y = top + height - point.precision * height;
      if (index === 0) curve.moveTo(x, top + height);
      curve.lineTo(x, y);
    });
    curve.lineTo(left + width, top + height);
    curve.closePath();
    curve.fillStyle = "rgba(22, 125, 131, 0.18)";
    curve.fill();

    curve.beginPath();
    ordered.forEach((point, index) => {
      const x = left + point.recall * width;
      const y = top + height - point.precision * height;
      if (index === 0) curve.moveTo(x, y);
      else curve.lineTo(x, y);
    });
    curve.strokeStyle = "#167d83";
    curve.lineWidth = 3;
    curve.stroke();

    const selectedX = left + selected.recall * width;
    const selectedY = top + height - selected.precision * height;
    curve.beginPath();
    curve.arc(selectedX, selectedY, 6, 0, Math.PI * 2);
    curve.fillStyle = "#e05d3f";
    curve.fill();
    curve.strokeStyle = "#ffffff";
    curve.lineWidth = 2;
    curve.stroke();

    curve.fillStyle = "#e05d3f";
    curve.font = "bold 11px sans-serif";
    curve.fillText("limiar atual", Math.min(selectedX + 9, 270), Math.max(selectedY - 8, 18));
  }

  function update() {
    const threshold = Number(thresholdInput.value) / 100;
    const data = metricsFor(threshold);
    thresholdValue.textContent = `${Math.round(threshold * 100)}%`;
    precisionValue.textContent = `${(data.precision * 100).toFixed(1)}%`;
    recallValue.textContent = `${(data.recall * 100).toFixed(1)}%`;
    countsValue.textContent = `TP: ${data.truePositives} | FP: ${data.falsePositives} | FN: ${data.falseNegatives}`;
    calculationValue.textContent =
      `Precisão = ${data.truePositives} / (${data.truePositives} + ${data.falsePositives}) | ` +
      `Recall = ${data.truePositives} / (${data.truePositives} + ${data.falseNegatives})`;
    apValue.textContent =
      `Neste exemplo, o AP aproximado da classe é ${(averagePrecision() * 100).toFixed(1)}%. ` +
      "A área azul reúne todos os limiares; o ponto vermelho representa apenas o limiar atual.";
    drawScene(data);
    drawCurve(data);
  }

  thresholdInput.addEventListener("input", update);
  update();
});