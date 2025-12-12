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

# Copio requirements (può stare in root oppure dentro tokenopt_site)
# Se tu hai requirements dentro tokenopt_site, cambia la riga COPY sotto.
COPY tokenopt_site/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r /app/requirements.txt

# Copio ENTRAMBE le repo dentro l'immagine
COPY tokenopt_site /app/tokenopt_site
COPY tokenopt_generator /app/tokenopt_generator

# Rendo importabile tokenopt_generator e tokenopt_site
ENV PYTHONPATH="/app/tokenopt_generator:/app/tokenopt_site"
