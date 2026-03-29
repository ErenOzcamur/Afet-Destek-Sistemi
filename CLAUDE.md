# ASTRO-RESQ — Claude Code Talimatları

## Proje Özeti
TUA Astro Hackathon 2026 için geliştirilen afet müdahale karar destek sistemi.
Streamlit tabanlı, uydu görüntüsü analizi + canlı sensör verisi + lojistik navigasyon içerir.

## Uygulamayı Çalıştırma
```bash
source .venv/bin/activate
streamlit run main.py
```

## Dosya Yapısı
| Dosya | Görev |
|---|---|
| `main.py` | Ana Streamlit uygulaması (~3500 satır), 5 sekme |
| `weather_sensor.py` | Meteorolojik sensör kararları, 79 il, Open-Meteo API |
| `ai_vision_analyzer.py` | Gemini / Anthropic görüntü analizi |
| `cems_service.py` | Copernicus EMS entegrasyonu (WFS mock fallback) |
| `planet_client.py` | Planet Labs uydu görüntüsü |
| `routing_engine.py` | OSMnx yol ağı hesapları |
| `seismic_data.py` | USGS / Kandilli deprem verisi |

## 5 Ana Sekme
1. **Meteorolojik Sensör** — Otonom karar motoru, risk kartları
2. **Canlı Sismik** — Gerçek zamanlı deprem verisi (USGS + Kandilli)
3. **Görüntü & Analiz** — Uydu görüntüsü + Gemini AI analizi
4. **Hava Durumu** — Meteorolojik erken uyarı, 112 bildirimi
5. **Lojistik Yollar** — Şehir içi (OSMnx) ve şehirlerarası (OSRM) navigasyon

## API Anahtarları
`.streamlit/secrets.toml` dosyasında saklanır — git'e commit etme.
- `planet.api_key` — Planet Labs
- `anthropic.api_key` — Claude Vision
- `gemini.api_key` — Google Gemini (ücretsiz: aistudio.google.com)
- `openweather.api_key` — OpenWeatherMap

## Önemli Teknik Notlar

### Streamlit Kuralları
- `@st.cache_data` dekoratörleri mutlaka modül üst seviyesinde tanımlanmalı — `with` bloğu veya tab içinde olmaz
- Harita etkileşiminde sayfa yenilenmesini önlemek için tüm `st_folium()` çağrılarında `returned_objects=[]` kullan
- Folium'da custom tile URL kullanırken `attr` boş string olamaz — en az `" "` (boşluk) olmalı
- Attribution'ı gizlemek için `attribution_control=False` Map parametresi kullan

### Routing Sistemi
- **Şehir içi**: OSMnx `graph_from_point` + NetworkX — gerçek yol ağı indirir, yavaş ama doğru
- **Şehirlerarası**: OSRM demo API (`router.project-osrm.org`) — gerçek karayolu geometrisi, hızlı
- OSRM response: `routes[0].geometry.coordinates` → `[lon, lat]` — folium için `(lat, lon)` çevir
- Hasar simülasyonu: rotayı ~40 segmente böl, random seed(42) ile kapalı/hasarlı ata

### Lojistik Tab — İl Yapısı
- `_LOJ_ILLER`: 81 ilin koordinatları (dict)
- `_gen_facilities(lat, lon, il)`: Her il için 15 tesis üretir (koordinat offset bazlı)
- Şehirlerarası modda hedef tesis, hedef ilin merkezinden üretilir

### Copernicus WFS
WFS endpoint'leri kalıcı olarak 404 dönüyor. Mock fallback doğru davranış — düzeltmeye gerek yok.

### AI Vision
- Default backend: `gemini` (`gemini-2.0-flash` modeli)
- `AIVisionAnalyzer(backend="gemini")` veya `backend="anthropic"`
- Model referansı: `self.GEMINI_MODEL` / `self.ANTHROPIC_MODEL` — `self.MODEL` yok

## Renk Kodları (Harita)
- `#4361ee` — Önerilen rota (mavi)
- `#f77f00` — Hasarlı yol (turuncu)
- `#e63946` — Kapalı yol (kırmızı)
- `#06d6a0` — Açık yol (yeşil)

## Bağımlılıklar Kurulumu
```bash
pip install -r requirements.txt
```
Kritik paketler: `streamlit`, `folium`, `streamlit-folium`, `osmnx`, `networkx`, `google-generativeai`, `anthropic`, `streamlit-js-eval`
