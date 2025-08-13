FROM python:3.11.9-slim

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements dosyasını kopyala ve pip güncelle
COPY requirements.txt .
RUN pip install --upgrade pip

# Büyük paketler için timeout 900 saniye (15dk)
RUN pip install --default-timeout=900 --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# GPU destekli PyTorch (CUDA 11.8)
RUN pip install --default-timeout=900 torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --upgrade

# SpaCy Türkçe modelini yükle
COPY data/tr_core_news_trf-1.0-py3-none-any.whl .
RUN pip install tr_core_news_trf-1.0-py3-none-any.whl

# Uygulama dosyalarını kopyala
COPY . .

# Log klasörü oluştur
RUN mkdir -p logs

# Port 8000 aç
EXPOSE 8000

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=60s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1


# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
