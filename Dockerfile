FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Seleziona quale requirements installare: dev (web+worker) oppure gpu
ARG APP_ENV=dev

# Copia requirements nella image
COPY requirements.base.txt /app/requirements.base.txt
COPY requirements.web.txt /app/requirements.web.txt
COPY requirements.worker.txt /app/requirements.worker.txt
COPY requirements.gpu.txt /app/requirements.gpu.txt

RUN pip install --upgrade pip wheel setuptools &&\
 if [ "$APP_ENV" = "gpu" ]; then \
    pip install -r /app/requirements.gpu.txt ; \
 else \
        pip install -r /app/requirements.web.txt && \
        pip install -r /app/requirements.worker.txt ; \
 fi


# Copio ENTRAMBE le repo dentro l'immagine
COPY tokenopt_site /app/tokenopt_site
COPY tokenopt_generator /app/tokenopt_generator

# Rendo importabile tokenopt_generator e tokenopt_site
ENV PYTHONPATH="/app/tokenopt_generator:/app/tokenopt_site"
