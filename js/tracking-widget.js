document.addEventListener("DOMContentLoaded", function () {
  const root = document.getElementById("tracking-widget");
  if (!root) return;

  const canvas = root.querySelector("[data-tracking-canvas]");
  const context = canvas.getContext("2d");
  const previousButton = root.querySelector("[data-tracking-previous]");
  const nextButton = root.querySelector("[data-tracking-next]");
  const playButton = root.querySelector("[data-tracking-play]");
  const modeInput = root.querySelector("[data-tracking-mode]");
  const occlusionInput = root.querySelector("[data-tracking-occlusion]");
  const frameOutput = root.querySelector("[data-tracking-frame]");
  const summaryOutput = root.querySelector("[data-tracking-summary]");
  const eventsOutput = root.querySelector("[data-tracking-events]");

  const frames = [
    [{ id: 1, x: 60 }, { id: 2, x: 440 }],
    [{ id: 1, x: 110 }, { id: 2, x: 390 }],
    [{ id: 1, x: 160 }, { id: 2, x: 340 }],
    [{ id: 1, x: 210 }, { id: 2, x: 290 }],
    [{ id: 1, x: 260 }, { id: 2, x: 240 }],
    [{ id: 1, x: 310 }, { id: 2, x: 190 }],
    [{ id: 1, x: 360 }, { id: 2, x: 140 }],
  ];
  const colors = { 1: "#e05d3f", 2: "#167d83" };
  let frameIndex = 0;
  let timer = null;

  function drawPerson(x, color) {
    context.fillStyle = color;
    context.beginPath();
    context.arc(x + 30, 74, 16, 0, Math.PI * 2);
    context.fill();
    context.fillRect(x + 17, 91, 26, 82);
    context.fillRect(x + 10, 169, 13, 56);
    context.fillRect(x + 37, 169, 13, 56);
  }

  function drawBox(person, faded) {
    const x = person.x;
    const y = 50;
    const width = 60;
    const height = 178;
    context.save();
    context.globalAlpha = faded ? 0.28 : 1;
    context.strokeStyle = colors[person.id];
    context.lineWidth = 3;
    if (faded) context.setLineDash([6, 5]);
    context.strokeRect(x, y, width, height);
    context.setLineDash([]);
    context.fillStyle = colors[person.id];
    context.fillRect(x, 28, 76, 20);
    context.fillStyle = "#ffffff";
    context.font = "bold 12px sans-serif";
    context.fillText(`ID ${person.id}`, x + 8, 43);
    context.restore();
  }

  function drawArrow(previous, current) {
    const fromX = previous.x + 30;
    const fromY = 142;
    const toX = current.x + 30;
    const toY = 142;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    context.save();
    context.strokeStyle = colors[current.id];
    context.fillStyle = colors[current.id];
    context.lineWidth = 2;
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(fromX, fromY);
    context.lineTo(toX, toY);
    context.stroke();
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(toX, toY);
    context.lineTo(toX - 9 * Math.cos(angle - Math.PI / 6), toY - 9 * Math.sin(angle - Math.PI / 6));
    context.lineTo(toX - 9 * Math.cos(angle + Math.PI / 6), toY - 9 * Math.sin(angle + Math.PI / 6));
    context.closePath();
    context.fill();
    context.restore();
  }

  function isOccluded(person) {
    return occlusionInput.checked && frameIndex === 3 && person.id === 2;
  }

  function drawScene() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#e8f0ee";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#b8d7de";
    context.fillRect(0, 0, canvas.width, 48);
    context.fillStyle = "#c7c1b2";
    context.fillRect(0, 228, canvas.width, 52);
    context.fillStyle = "#f7f1df";
    context.fillRect(268, 20, 64, 208);
    context.fillStyle = "#536b75";
    context.fillRect(276, 28, 48, 200);
    context.strokeStyle = "#f1a208";
    context.lineWidth = 3;
    context.setLineDash([8, 6]);
    context.beginPath();
    context.moveTo(300, 0);
    context.lineTo(300, canvas.height);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#163038";
    context.font = "12px sans-serif";
    context.fillText("linha de contagem", 307, 18);
  }

  function updateText(current) {
    const tracking = modeInput.checked;
    const occluded = current.some(isOccluded);
    frameOutput.textContent = `Quadro ${frameIndex + 1} de ${frames.length}`;
    if (!tracking) {
      summaryOutput.textContent = "Detecção: cada quadro mostra caixas novas. Ainda não sabemos qual caixa pertence ao mesmo objeto no quadro anterior.";
      eventsOutput.textContent = "Não há IDs persistentes para usar em uma contagem.";
      return;
    }
    if (occluded) {
      summaryOutput.textContent = "Oclusão: a pessoa azul não foi detectada neste quadro. O rastreador perdeu temporariamente a evidência visual para associar seu ID.";
      eventsOutput.textContent = "No próximo quadro, o ID 2 pode ser recuperado ou pode mudar. Tracking é uma estimativa.";
      return;
    }
    const crossed = current.filter((person) => person.x + 30 >= 300 && frames[frameIndex - 1]?.find((previous) => previous.id === person.id)?.x + 30 < 300);
    summaryOutput.textContent = "Tracking: as setas conectam uma caixa atual à caixa do mesmo ID no quadro anterior.";
    eventsOutput.textContent = crossed.length
      ? `${crossed.map((person) => `ID ${person.id}`).join(" e ")} cruzou a linha: registre um evento, não uma nova detecção por quadro.`
      : "Nenhum novo ID cruzou a linha neste quadro.";
  }

  function render() {
    const current = frames[frameIndex];
    const tracking = modeInput.checked;
    drawScene();

    if (tracking && frameIndex > 0) {
      frames[frameIndex - 1].forEach((person) => {
        if (!isOccluded(person)) drawBox(person, true);
      });
    }

    current.forEach((person) => {
      if (isOccluded(person)) return;
      drawPerson(person.x, colors[person.id]);
      if (tracking) {
        if (frameIndex > 0) {
          const previous = frames[frameIndex - 1].find((item) => item.id === person.id);
          if (previous) drawArrow(previous, person);
        }
        drawBox(person, false);
      } else {
        context.strokeStyle = "#576b7a";
        context.lineWidth = 3;
        context.strokeRect(person.x, 50, 60, 178);
        context.fillStyle = "#576b7a";
        context.fillRect(person.x, 28, 98, 20);
        context.fillStyle = "#ffffff";
        context.font = "bold 12px sans-serif";
        context.fillText("pessoa 91%", person.x + 6, 43);
      }
    });

    if (occlusionInput.checked && frameIndex === 3) {
      context.fillStyle = "rgba(22, 48, 56, 0.72)";
      context.fillRect(270, 25, 60, 205);
      context.fillStyle = "#ffffff";
      context.font = "bold 12px sans-serif";
      context.fillText("oclusão", 274, 126);
    }
    updateText(current);
  }

  function stop() {
    if (timer) window.clearInterval(timer);
    timer = null;
    playButton.textContent = "Reproduzir";
  }

  function step(change) {
    frameIndex = (frameIndex + change + frames.length) % frames.length;
    render();
  }

  previousButton.addEventListener("click", () => { stop(); step(-1); });
  nextButton.addEventListener("click", () => { stop(); step(1); });
  playButton.addEventListener("click", () => {
    if (timer) {
      stop();
      return;
    }
    playButton.textContent = "Pausar";
    timer = window.setInterval(() => step(1), 850);
  });
  modeInput.addEventListener("change", render);
  occlusionInput.addEventListener("change", render);
  render();
});
