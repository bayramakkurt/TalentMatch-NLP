# 🎯 TalentMatch NLP - CV Analizi ve İş Eşleştirme Sistemi

Bu proje, PDF/DOCX formatındaki CV'leri analiz ederek iş ilanlarıyla otomatik eşleştirme yapan gelişmiş bir NLP uygulamasıdır.

## 🚀 Özellikler

### ✅ CV İşleme
- **Dosya Formatları**: PDF, DOC, DOCX desteği
- **Otomatik Çıkarım**: Ad-soyad, eğitim, deneyim, beceriler, iletişim bilgileri
- **AI Özetleme**: Türkçe T5 modeli ile CV özetleri
- **MongoDB GridFS**: Güvenli dosya saklama

### ✅ İş Eşleştirme
- **FAISS Vektör Arama**: Hızlı ve doğru benzerlik analizi
- **Sentence Transformers**: Gelişmiş NLP embeddings
- **Eşleşme Skoru**: Yüzdelik dilimde uyumluluk
- **Eksik Beceri Analizi**: Aday geliştirme alanları

### ✅ Bildirim Sistemi
- **E-posta Entegrasyonu**: SMTP ile otomatik bildirimler
- **Responsive Template**: HTML e-posta şablonları
- **Manuel Kontrol**: Admin onayı ile gönderim
- **Toplu İşlem**: Çoklu aday bildirimi

### ✅ Admin Paneli
- **Streamlit Arayüz**: Kullanıcı dostu web paneli
- **CV Yönetimi**: Görüntüleme, silme, detaylar
- **İş İlanı Yönetimi**: CRUD işlemleri
- **İstatistikler**: Kapsamlı raporlama

## 🏗️ Sistem Mimarisi

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│                 │    │                 │    │                 │
│ • Streamlit     │◄──►│ • FastAPI       │◄──►│ • MongoDB       │
│ • Admin Panel   │    │ • CV Parser     │    │ • GridFS        │
│ • Web UI        │    │ • NLP Engine    │    │ • Collections   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   AI Services   │
                    │                 │
                    │ • spaCy NER     │
                    │ • HuggingFace   │
                    │ • FAISS Search  │
                    │ • T5 Summary    │
                    └─────────────────┘
```

## 📦 Kurulum

### Docker ile Hızlı Başlangıç (Önerilen)

```bash
# Projeyi klonla
git clone <repository-url>
cd talentmatch-nlp

# Environment dosyasını oluştur
cp .env.template .env
# .env dosyasını düzenle (SMTP ayarları vs.)

# Docker container'ları başlat
docker-compose up -d

# Servisler hazır!
# API: http://localhost:8000
# Admin Panel: http://localhost:8501
# MongoDB: localhost:27017
```

### Manuel Kurulum

```bash
# Python sanal ortamı oluştur
python -m venv talentmatch_env
source talentmatch_env/bin/activate  # Linux/Mac
# talentmatch_env\Scripts\activate   # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# spaCy Türkçe modelini indir
python -m spacy download tr_core_news_sm

# MongoDB'yi başlat (ayrı terminal)
mongod

# API'yi başlat
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Admin paneli başlat (ayrı terminal)
streamlit run admin_panel.py --server.port 8501
```

## ⚙️ Konfigürasyon

### .env Dosyası

```bash
# MongoDB
MONGODB_URI=mongodb://localhost:27017/

# SMTP E-posta
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Log
LOG_LEVEL=INFO
```

### SMTP Ayarları

Gmail için uygulama parolası oluşturun:
1. Gmail → Ayarlar → Güvenlik
2. 2-Adımlı Doğrulama'yı etkinleştirin
3. Uygulama parolaları → Mail seçin
4. Oluşturulan parolayı `SMTP_PASSWORD`'e yazın

## 📚 API Dokümantasyonu

### Temel Endpoint'ler

```bash
# API dokümantasyonu
GET /docs

# Sistem durumu
GET /health

# CV yükleme
POST /upload-cv
Content-Type: multipart/form-data

# İş ilanı oluşturma
POST /job-posting
{
    "title": "Python Developer",
    "description": "...",
    "requirements": ["Python", "Django"],
    "location": "İstanbul",
    "company": "ABC Tech"
}

# Eşleştirme yapma
POST /match-candidates/{job_id}

# Bildirim gönderme
POST /send-notifications
{
    "job_id": "job_id_here",
    "candidate_ids": ["candidate1", "candidate2"]
}
```

### Response Formatları

```json
{
    "status": "success",
    "message": "İşlem başarılı",
    "data": { ... },
    "count": 5
}
```

## 🧪 Test

```bash
# API testleri
python -m pytest tests/ -v

# Test coverage
python -m pytest --cov=. tests/

# Belirli endpoint testi
curl -X GET http://localhost:8000/health
```

## 🔧 Geliştirme

### Proje Yapısı

```
talentmatch-nlp/
├── main.py              # FastAPI ana uygulama
├── cv_parse.py          # CV işleme modülleri
├── mongodb.py           # Veritabanı işlemleri
├── vector.py            # FAISS vektör eşleştirme
├── notify.py            # E-posta bildirimi
├── admin_panel.py       # Streamlit admin paneli
├── requirements.txt     # Python bağımlılıkları
├── docker-compose.yml   # Docker orkestrasyonu
├── Dockerfile          # API container
├── Dockerfile.admin    # Admin panel container
└── .env.template       # Environment şablonu
```

### Yeni Özellik Ekleme

1. **API Endpoint**: `main.py`'ye yeni route ekle
2. **Veri İşleme**: İlgili modüle fonksiyon ekle
3. **Admin Panel**: `admin_panel.py`'ye UI ekle
4. **Test**: Test dosyası oluştur

### Kod Kalitesi

```bash
# Code formatting
black .

# Import sorting
isort .

# Linting
flake8 .

# Type checking
mypy .
```

## 📊 Performans

### Optimizasyon İpuçları

1. **FAISS İndeks**: Büyük veri setleri için IVF indeks kullan
2. **MongoDB**: Compound index'ler oluştur
3. **Caching**: Redis ile API yanıt cache'i
4. **Batch Processing**: Çoklu CV işleme için queue sistemi


⭐ **Bu projeyi beğendiyseniz yıldız verin!** ⭐