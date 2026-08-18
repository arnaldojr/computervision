document.addEventListener("DOMContentLoaded", function () {
  const root = document.getElementById("vit-patches-widget");
  if (!root) return;

  const inputCanvas = root.querySelector("[data-vit-input]");
  const patchCanvas = root.querySelector("[data-vit-patches]");
  const tokenCanvas = root.querySelector("[data-vit-tokens]");
  const inputContext = inputCanvas.getContext("2d");
  const patchContext = patchCanvas.getContext("2d");
  const tokenContext = tokenCanvas.getContext("2d");
  const patchSize = root.querySelector("[data-vit-patch-size]");
  const patchValue = root.querySelector("[data-vit-patch-value]");
  const patchCount = root.querySelector("[data-vit-patch-count]");
  const imageShape = root.querySelector("[data-vit-image-shape]");
  const tokenShape = root.querySelector("[data-vit-token-shape]");
  const selectedPatch = root.querySelector("[data-vit-selected]");
  const selectedCoordinates = root.querySelector("[data-vit-coordinates]");
  const status = root.querySelector("[data-vit-status]");
  const exampleButtons = root.querySelectorAll("[data-vit-example]");
  const clearButton = root.querySelector("[data-vit-clear]");

  const size = 16;
  let pixels = new Float32Array(size * size * 3);
  let selected = 0;
  let drawing = false;

  function pixelIndex(x, y) {
    return (y * size + x) * 3;
  }

  function setPixel(x, y, red, green, blue) {
    if (x < 0 || x >= size || y < 0 || y >= size) return;
    const index = pixelIndex(x, y);
    pixels[index] = red;
    pixels[index + 1] = green;
    pixels[index + 2] = blue;
  }

  function fillPixels(red, green, blue) {
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) setPixel(x, y, red, green, blue);
    }
  }

  function loadExample(name) {
    fillPixels(0.96, 0.95, 0.9);
    if (name === "diagonal") {
      for (let i = 1; i < 15; i += 1) {
        setPixel(i, i, 0.1, 0.38, 0.45);
        setPixel(i, i - 1, 0.18, 0.58, 0.56);
        setPixel(i, i + 1, 0.18, 0.58, 0.56);
      }
    } else if (name === "object") {
      for (let y = 4; y < 12; y += 1) {
        for (let x = 5; x < 11; x += 1) setPixel(x, y, 0.92, 0.42, 0.2);
      }
      for (let x = 4; x < 12; x += 1) {
        setPixel(x, 3, 0.1, 0.38, 0.45);
        setPixel(x, 12, 0.1, 0.38, 0.45);
      }
      for (let y = 3; y < 13; y += 1) {
        setPixel(4, y, 0.1, 0.38, 0.45);
        setPixel(11, y, 0.1, 0.38, 0.45);
      }
    }
    drawAll();
  }

  function drawInput() {
    const cell = inputCanvas.width / size;
    inputContext.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const index = pixelIndex(x, y);
        const red = Math.round(pixels[index] * 255);
        const green = Math.round(pixels[index + 1] * 255);
        const blue = Math.round(pixels[index + 2] * 255);
        inputContext.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        inputContext.fillRect(x * cell, y * cell, cell, cell);
      }
    }
    inputContext.strokeStyle = "rgba(22, 48, 56, 0.12)";
    inputContext.lineWidth = 1;
    for (let i = 0; i <= size; i += 1) {
      inputContext.beginPath();
      inputContext.moveTo(i * cell, 0);
      inputContext.lineTo(i * cell, inputCanvas.height);
      inputContext.stroke();
      inputContext.beginPath();
      inputContext.moveTo(0, i * cell);
      inputContext.lineTo(inputCanvas.width, i * cell);
      inputContext.stroke();
    }
  }

  function drawPatches() {
    const countPerSide = Number(patchSize.value);
    const cell = patchCanvas.width / size;
    patchContext.clearRect(0, 0, patchCanvas.width, patchCanvas.height);
    patchContext.drawImage(inputCanvas, 0, 0);
    patchContext.lineWidth = 2;
    for (let y = 0; y < countPerSide; y += 1) {
      for (let x = 0; x < countPerSide; x += 1) {
        const index = y * countPerSide + x;
        patchContext.strokeStyle = index === selected ? "#e05d3f" : "rgba(22, 48, 56, 0.5)";
        patchContext.strokeRect(x * cell * patchSize.value, y * cell * patchSize.value, cell * patchSize.value, cell * patchSize.value);
      }
    }
  }

  function drawTokens() {
    const countPerSide = Number(patchSize.value);
    const count = countPerSide * countPerSide;
    const gap = 5;
    const tokenWidth = (tokenCanvas.width - gap * (count + 1)) / count;
    tokenContext.clearRect(0, 0, tokenCanvas.width, tokenCanvas.height);
    tokenContext.font = "12px sans-serif";
    tokenContext.textAlign = "center";
    for (let index = 0; index < count; index += 1) {
      const x = gap + index * (tokenWidth + gap);
      tokenContext.fillStyle = index === selected ? "#e05d3f" : "#167d83";
      tokenContext.fillRect(x, 24, tokenWidth, 42);
      tokenContext.fillStyle = "#ffffff";
      tokenContext.fillText(`t${index}`, x + tokenWidth / 2, 50);
    }
    tokenContext.fillStyle = "#163038";
    tokenContext.fillText("ordem de leitura: esquerda → direita, linha a linha", tokenCanvas.width / 2, 91);
  }

  function drawAll() {
    const countPerSide = Number(patchSize.value);
    const count = countPerSide * countPerSide;
    selected = Math.min(selected, count - 1);
    drawInput();
    drawPatches();
    drawTokens();
    patchValue.textContent = `${patchSize.value} × ${patchSize.value} pixels`;
    patchCount.textContent = `${count} patches`;
    imageShape.textContent = `3 × ${size} × ${size}`;
    tokenShape.textContent = `${count} × dimensão do embedding`;
    const row = Math.floor(selected / countPerSide);
    const column = selected % countPerSide;
    selectedPatch.textContent = `patch ${selected} → token t${selected}`;
    selectedCoordinates.textContent = `linha ${row}, coluna ${column}`;
    status.textContent = "Clique em um patch para acompanhar sua posição na sequência.";
  }

  function selectFromEvent(event) {
    const rect = patchCanvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * size);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * size);
    const countPerSide = Number(patchSize.value);
    const patchX = Math.floor(x / Number(patchSize.value));
    const patchY = Math.floor(y / Number(patchSize.value));
    if (patchX >= 0 && patchX < countPerSide && patchY >= 0 && patchY < countPerSide) {
      selected = patchY * countPerSide + patchX;
      drawAll();
    }
  }

  function paint(event) {
    const rect = inputCanvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * size);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * size);
    setPixel(x, y, 0.1, 0.38, 0.45);
    drawAll();
  }

  patchSize.addEventListener("input", () => {
    selected = 0;
    drawAll();
  });
  patchCanvas.addEventListener("click", selectFromEvent);
  inputCanvas.addEventListener("pointerdown", (event) => {
    drawing = true;
    inputCanvas.setPointerCapture(event.pointerId);
    paint(event);
  });
  inputCanvas.addEventListener("pointermove", (event) => {
    if (drawing) paint(event);
  });
  inputCanvas.addEventListener("pointerup", () => { drawing = false; });
  clearButton.addEventListener("click", () => { fillPixels(0.96, 0.95, 0.9); drawAll(); });
  exampleButtons.forEach((button) => button.addEventListener("click", () => loadExample(button.dataset.vitExample)));

  loadExample("object");
});
