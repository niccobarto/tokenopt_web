# Immagine base Python 3.11 in versione minimale
FROM python:3.11-slim

# Disabilita la generazione di file .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Disabilita il buffering dell'output (log immediati)
ENV PYTHONUNBUFFERED=1

# Imposta la directory di lavoro principale
WORKDIR /app

# Aggiorna apt e installa dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Toolchain per compilare estensioni Python native
    build-essential \
    # Git per dipendenze o operazioni su repository
    git \
    # Curl per richieste HTTP e debug
    curl \
    # Libreria richiesta da molte dipendenze di image processing
    libglib2.0-0 \
    # Libreria OpenGL di base (OpenCV, PIL, ecc.)
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Build argument per distinguere ambiente dev o gpu
ARG APP_ENV=dev

# Copia i file requirements nell'immagine
COPY requirements.base.txt /app/requirements.base.txt
COPY requirements.web.txt /app/requirements.web.txt
COPY requirements.worker.txt /app/requirements.worker.txt
COPY requirements.gpu.txt /app/requirements.gpu.txt

# Aggiorna pip e installa i requirements in base all'ambiente
RUN if [ "$APP_ENV" = "gpu" ]; then \
        pip install -r /app/requirements.gpu.txt ; \
    else \
        pip install -r /app/requirements.web.txt && \
        pip install -r /app/requirements.worker.txt ; \
    fi

# Copia il codice del progetto Django nell'immagine
COPY tokenopt_site /app/tokenopt_site

# Copia il codice del generatore Token-Opt nell'immagine
COPY tokenopt_generator /app/tokenopt_generator

# Imposta il PYTHONPATH per consentire import tra le due repository
ENV PYTHONPATH="/app/tokenopt_generator:/app/tokenopt_site"
