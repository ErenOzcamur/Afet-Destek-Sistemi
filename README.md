# 🛰️ TUA Astro DSS — Afet Sonrası Hızlı Hasar Tespiti & Lojistik Optimizasyonu

> **TUA Astro Hackathon** — 48 Saatlik Sprint  
> Yerli uydu verileri (İMECE, GÖKTÜRK-1) ile çalışan Karar Destek Sistemi

---

## 📁 Proje Yapısı

```
dss_project/
├── main.py              # Streamlit ana uygulama & UI
├── config.py            # Merkezi konfigürasyon (dataclass)
├── utils.py             # I/O, preprocessing pipeline, loglama
├── cv_engine.py         # Computer Vision hasar tespiti motoru
├── routing_engine.py    # OSMnx lojistik rota optimizasyonu
├── map_builder.py       # Folium harita görselleştirme
├── requirements.txt     # Python bağımlılıkları
├── assets/              # Statik dosyalar
├── logs/                # Otomatik oluşturulan log dosyaları
└── README.md            # Bu dosya
```

---

## 🚀 Kurulum & Çalıştırma

```bash
# 1. Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. Uygulamayı başlat
streamlit run main.py
```

---

## 🧠 Teknik Mimari

### Computer Vision Pipeline
```
GeoTIFF/PNG → Normalize → Denoise → CLAHE → [Align] → SSIM + AbsDiff
   → Binary Mask (Otsu + Morfoloji) → Kontur Analizi → Hasar Raporu
```

### Lojistik Optimizasyon
```
Hasar Koordinatları → OSMnx Yol Ağı → Engel İşaretleme (Buffer)
   → Dijkstra Güvenli Rota → Alternatif Rotalar (k-shortest)
```

---

## 📊 Yerli Uydu Verisi Avantajı

| Kaynak          | GSD     | Precision  | Recall    | F1-Score  |
|-----------------|---------|------------|-----------|-----------|
| İMECE (<1m)     | ~0.8m   | 0.85-0.92  | 0.80-0.88 | 0.82-0.90 |
| GÖKTÜRK-1       | ~2.5m   | 0.78-0.85  | 0.72-0.80 | 0.75-0.82 |
| Sentinel-2      | ~10m    | 0.55-0.65  | 0.45-0.55 | 0.50-0.60 |

---

## 🔌 SAM Entegrasyonu (Opsiyonel)

```python
# GPU gereklidir
pip install segment-anything
# SAM checkpoint indir (vit_b ~375MB)

detector = DamageDetector()
report = detector.detect_damage_with_sam(before, after)
```

---

## 📝 Lisans

Hackathon projesidir. Afet müdahale amacıyla serbestçe kullanılabilir.
