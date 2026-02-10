# Ferramentas e Instalação — Visão Computacional

!!! info "Escolha seu caminho"
    Você pode trabalhar **localmente (recomendado)** ou usar **Google Colab** (sem instalação).

[Baixar Python](https://www.python.org/downloads/){ .md-button .md-button-primary }
[VS Code](https://code.visualstudio.com/){ .md-button }
[Google Colab](https://colab.research.google.com/){ .md-button }

---

## 1) Opção A — Instalação local (recomendada)

### 1.1 Python

!!! tip "Versão recomendada"
    Use **Python 3.11 ou 3.12** para melhor compatibilidade com o ecossistema do curso.

=== "Windows"
    1. Baixe o instalador em: `https://www.python.org/downloads/`
    2. Durante a instalação, marque **Add Python to PATH**.
    3. Verifique no terminal:
        ```bash
        python --version
        ```

=== "macOS"
    **Homebrew (recomendado, se você já usa):**
    ```bash
    brew install python
    python3 --version
    ```

=== "Linux (Ubuntu/Debian)"
    ```bash
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip
    python3 --version
    ```

---

### 1.2 IDE (ambiente de desenvolvimento)

!!! recommended "Recomendado para o curso"
    - **Visual Studio Code** (principal)
    - **JupyterLab** (notebooks)
    - **Google Colab** (alternativa em nuvem)

**VS Code (passo a passo)**

1. Instale: https://code.visualstudio.com/
2. Abra o VS Code e instale a extensão **Python** (Microsoft).
3. (Opcional) Instale também **Jupyter** (para notebooks).

---

### 1.3 Crie um ambiente virtual (venv)

!!! info "Por que usar venv?"
    Mantém as dependências do curso isoladas do restante do seu sistema.

=== "macOS / Linux"
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows (PowerShell)"
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

!!! warning "PowerShell bloqueando ativação?"
    Execute uma vez:
    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    ```

---

### 1.4 Instale as bibliotecas do curso

!!! info "Pacote mínimo do curso (primeiras aulas)"
    ```bash
    pip install numpy matplotlib scikit-learn jupyterlab opencv-python
    ```

!!! note "Deep Learning (quando o curso entrar em redes neurais)"
    ```bash
    pip install tensorflow
    ```

=== "macOS Apple Silicon (opcional: aceleração por GPU/Metal)"
    ```bash
    pip install tensorflow-metal
    ```

---

### 1.5 Teste rápido do ambiente (sanity check)

Crie um arquivo `check_env.py` com o conteúdo abaixo:

```python
import sys
import cv2
import numpy as np
import matplotlib
import sklearn

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OK ✅")
```

Rode no terminal (com o `.venv` ativado):
```bash
python check_env.py
```

!!! success "Se apareceu OK ✅"
    Seu ambiente está pronto para as aulas práticas.

---

## 2) Opção B — Google Colab (sem instalação)

!!! tip "Quando usar"
    Use o Colab se você estiver sem permissão de instalar coisas no computador ou se precisar de GPU.

1. Acesse: https://colab.research.google.com/
2. Crie um notebook e rode (se necessário):
    ```python
    !pip install opencv-python
    ```

!!! note "Observação"
    No Colab, **NumPy/Matplotlib** geralmente já vêm instalados.

---

## 3) Abrir o JupyterLab

Com o ambiente virtual ativado:
```bash
jupyter lab
```

