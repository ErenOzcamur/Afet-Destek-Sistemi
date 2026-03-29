# ============================================================
# weather_sensor.py - Otonom Optik/SAR Sensör Karar Motoru
# TUA Astro Hackathon DSS — Open-Meteo Canlı Hava API'si
# ============================================================

"""
Anlık hava durumunu Open-Meteo'dan çeker (ücretsiz, API key gerekmez)
ve bulut örtüsüne göre otonom uydu sensör kararı üretir.

Karar Mantığı:
  Bulut ≤ 20% → İMECE (Optik)          — görüş mükemmel
  Bulut 21-60% → Planet SkySat (Optik)  — kısmi görüş, ticari optik
  Bulut > 60%  → Göktürk-2 (SAR Radar) — optik imkansız, radar zorunlu
  Yağış faktörü: precipitation * 10 → efektif bulutluluğa eklenir

Neden önemli:
  Optik uydular bulut altını göremez. Afet anında hava koşulları
  kötüyse SAR (Sentetik Açıklıklı Radar) tek seçenektir.
  Bu modül jüriye sistemin uydu fiziğini bildiğini kanıtlar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from loguru import logger


# ── Türkiye Afet Bölgeleri ───────────────────────────────────

DISASTER_REGIONS: Dict[str, Tuple[float, float]] = {
    # Marmara
    "İstanbul":           (41.0082, 28.9784),
    "Bursa":              (40.1826, 29.0665),
    "Kocaeli":            (40.7654, 29.9408),
    "Tekirdağ":           (40.9781, 27.5115),
    "Edirne":             (41.6818, 26.5623),
    "Balıkesir":          (39.6484, 27.8826),
    "Çanakkale":          (40.1553, 26.4142),
    "Sakarya":            (40.6940, 30.4358),
    "Yalova":             (40.6500, 29.2667),
    "Kırklareli":         (41.7351, 27.2253),
    # Ege
    "İzmir":              (38.4189, 27.1287),
    "Manisa":             (38.6191, 27.4289),
    "Aydın":              (37.8560, 27.8416),
    "Muğla":              (37.2153, 28.3636),
    "Denizli":            (37.7765, 29.0864),
    "Afyonkarahisar":     (38.7637, 30.5456),
    "Uşak":               (38.6823, 29.4082),
    "Kütahya":            (39.4242, 29.9834),
    # Akdeniz
    "Antalya":            (36.8969, 30.7133),
    "Mersin":             (36.8121, 34.6415),
    "Adana":              (37.0000, 35.3213),
    "Hatay (Antakya)":    (36.2021, 36.1603),
    "Isparta":            (37.7648, 30.5566),
    "Burdur":             (37.7265, 30.2903),
    "Osmaniye":           (37.0742, 36.2476),
    "Kahramanmaraş":      (37.5858, 36.9371),
    # İç Anadolu
    "Ankara":             (39.9334, 32.8597),
    "Konya":              (37.8747, 32.4932),
    "Kayseri":            (38.7312, 35.4787),
    "Sivas":              (39.7477, 37.0179),
    "Nevşehir":           (38.6939, 34.6857),
    "Aksaray":            (38.3687, 34.0370),
    "Niğde":              (37.9667, 34.6833),
    "Eskişehir":          (39.7767, 30.5206),
    "Çankırı":            (40.6013, 33.6134),
    "Kırıkkale":          (39.8456, 33.5153),
    "Kırşehir":           (39.1425, 34.1709),
    "Yozgat":             (39.8181, 34.8147),
    "Karaman":            (37.1833, 33.2167),
    # Karadeniz
    "Trabzon":            (41.0015, 39.7178),
    "Samsun":             (41.2867, 36.3300),
    "Zonguldak":          (41.4564, 31.7987),
    "Giresun":            (40.9128, 38.3895),
    "Rize":               (41.0201, 40.5234),
    "Ordu":               (40.9862, 37.8797),
    "Sinop":              (42.0231, 35.1531),
    "Kastamonu":          (41.3887, 33.7827),
    "Bartın":             (41.6344, 32.3375),
    "Karabük":            (41.2061, 32.6204),
    "Düzce":              (40.8439, 31.1565),
    "Bolu":               (40.7359, 31.6060),
    "Artvin":             (41.1828, 41.8183),
    "Amasya":             (40.6499, 35.8353),
    "Tokat":              (40.3167, 36.5544),
    "Çorum":              (40.5489, 34.9533),
    "Gümüşhane":          (40.4608, 39.4814),
    "Bayburt":            (40.2552, 40.2249),
    # Doğu Anadolu
    "Erzurum":            (39.9055, 41.2658),
    "Malatya":            (38.3552, 38.3095),
    "Elazığ":             (38.6748, 39.2264),
    "Diyarbakır":         (37.9144, 40.2306),
    "Van":                (38.4891, 43.4089),
    "Bingöl":             (38.8854, 40.4981),
    "Erzincan":           (39.7500, 39.5000),
    "Ağrı":               (39.7191, 43.0503),
    "Kars":               (40.6013, 43.0975),
    "Iğdır":              (39.9167, 44.0333),
    "Muş":                (38.9462, 41.7539),
    "Bitlis":             (38.3938, 42.1232),
    "Hakkari":            (37.5744, 43.7408),
    "Tunceli":            (39.1079, 39.5480),
    # Güneydoğu Anadolu
    "Adıyaman":           (37.7648, 38.2786),
    "Gaziantep":          (37.0662, 37.3833),
    "Şanlıurfa":          (37.1591, 38.7969),
    "Mardin":             (37.3212, 40.7245),
    "Batman":             (37.8812, 41.1351),
    "Siirt":              (37.9333, 41.9500),
    "Şırnak":             (37.5181, 42.4618),
    "Kilis":              (36.7184, 37.1212),
}

OPENMETEO_URL   = "https://api.open-meteo.com/v1/forecast"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# Uydu profilleri
SAT_PROFILES = {
    "İMECE":         {"type": "OPTICAL", "res_m": 0.9,  "flag": "🛰️",  "color": "#06d6a0"},
    "Göktürk-2":     {"type": "SAR",     "res_m": 2.5,  "flag": "📡",  "color": "#e63946"},
    "Planet SkySat": {"type": "OPTICAL", "res_m": 0.5,  "flag": "🛰️",  "color": "#f77f00"},
    "Sentinel-1":    {"type": "SAR",     "res_m": 10.0, "flag": "📡",  "color": "#adb5bd"},
}


# ── Dataclass'lar ────────────────────────────────────────────

@dataclass
class WeatherReading:
    """Ham hava durumu verisi — Open-Meteo ve/veya OpenWeatherMap."""
    lat:           float
    lon:           float
    cloud_cover:   int        # 0–100 % (iki kaynak ortalaması alınır)
    temperature:   float      # °C
    wind_speed:    float      # km/h
    precipitation: float      # mm (son 1 saat)
    visibility:    int        # metre

    # Kaynak bilgisi (UI'da gösterim için)
    source:        str = "Open-Meteo"          # "Open-Meteo" | "OWM" | "Fused"
    owm_cloud:     Optional[int] = None        # OWM ham bulut değeri
    omm_cloud:     Optional[int] = None        # Open-Meteo ham bulut değeri
    weather_desc:  str = ""                    # OWM "açık gökyüzü" vb.


@dataclass
class SensorDecision:
    """Otonom sensör seçim kararı."""
    region:         str
    weather:        WeatherReading

    # Karar
    satellite:      str        # "İMECE" / "Göktürk-2" / "Planet SkySat"
    sensor_type:    str        # "OPTICAL" | "SAR"
    status:         str        # "OPEN" | "PARTIAL" | "CLOSED"
    confidence:     str        # "HIGH" | "MEDIUM" | "LOW"
    decision_text:  str        # Jüriye yönelik kısa açıklama
    decision_long:  str        # Tam teknik gerekçe

    # Efektif bulutluluk (yağış dahil)
    effective_cloud: int

    @property
    def profile(self) -> Dict:
        return SAT_PROFILES.get(self.satellite, {})

    @property
    def status_color(self) -> str:
        return {"OPEN": "#06d6a0", "PARTIAL": "#f77f00", "CLOSED": "#e63946"}.get(
            self.status, "#adb5bd"
        )

    @property
    def status_label_tr(self) -> str:
        return {
            "OPEN":    "Görüş Açık: Optik Aktif",
            "PARTIAL": "Kısmi Görüş: Ticari Optik",
            "CLOSED":  "Görüş Kapalı: SAR Radar Aktif",
        }.get(self.status, self.status)


# ── Ana Fonksiyonlar ─────────────────────────────────────────

def _fetch_openmeteo(lat: float, lon: float) -> Dict:
    """Open-Meteo API'sinden ham veri çeker. Ücretsiz, key gerekmez."""
    r = requests.get(
        OPENMETEO_URL,
        params={
            "latitude":      lat,
            "longitude":     lon,
            "current":       [
                "cloud_cover", "temperature_2m",
                "wind_speed_10m", "precipitation",
            ],
            "forecast_days": 1,
            "timezone":      "Europe/Istanbul",
        },
        timeout=10,
    )
    r.raise_for_status()
    cur = r.json().get("current", {})
    return {
        "cloud_cover":   int(cur.get("cloud_cover", -1)),
        "temperature":   float(cur.get("temperature_2m", 0.0)),
        "wind_speed":    float(cur.get("wind_speed_10m", 0.0)),
        "precipitation": float(cur.get("precipitation", 0.0)),
    }


def _fetch_openweather(lat: float, lon: float, api_key: str) -> Dict:
    """OpenWeatherMap Current Weather API'sinden veri çeker."""
    r = requests.get(
        OPENWEATHER_URL,
        params={
            "lat":   lat,
            "lon":   lon,
            "appid": api_key,
            "units": "metric",
            "lang":  "tr",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "cloud_cover":   int(data.get("clouds", {}).get("all", -1)),
        "temperature":   float(data.get("main", {}).get("temp", 0.0)),
        "wind_speed":    float(data.get("wind", {}).get("speed", 0.0)) * 3.6,  # m/s → km/h
        "precipitation": float(
            data.get("rain", {}).get("1h", 0.0) or
            data.get("snow", {}).get("1h", 0.0)
        ),
        "visibility":    int(data.get("visibility", 10000)),
        "description":   data.get("weather", [{}])[0].get("description", ""),
    }


def fetch_weather(region: str, owm_api_key: Optional[str] = None) -> WeatherReading:
    """
    Anlık hava durumunu çeker.

    Strateji (dual-source fusion):
      1. Open-Meteo çekilir (her zaman, ücretsiz)
      2. owm_api_key varsa OpenWeatherMap da çekilir
      3. İkisi de başarılıysa bulut örtüsü ortalaması alınır (daha güvenilir)
      4. Yalnızca biri başarılıysa o kullanılır

    Args:
        region     : DISASTER_REGIONS anahtarı
        owm_api_key: OpenWeatherMap API anahtarı (opsiyonel)

    Returns:
        WeatherReading

    Raises:
        ValueError: Bilinmeyen bölge
        RuntimeError: Her iki API de başarısız
    """
    coords = DISASTER_REGIONS.get(region)
    if not coords:
        raise ValueError(f"Bilinmeyen bölge: {region}. Geçerli: {list(DISASTER_REGIONS)}")
    lat, lon = coords

    # ── Open-Meteo (birincil) ─────────────────────────────────
    omm_data: Optional[Dict] = None
    try:
        omm_data = _fetch_openmeteo(lat, lon)
        logger.info(f"Open-Meteo: bulut={omm_data['cloud_cover']}% — {region}")
    except Exception as e:
        logger.warning(f"Open-Meteo hatası: {e}")

    # ── OpenWeatherMap (ikincil, opsiyonel) ───────────────────
    owm_data: Optional[Dict] = None
    if owm_api_key:
        try:
            owm_data = _fetch_openweather(lat, lon, owm_api_key)
            logger.info(f"OWM: bulut={owm_data['cloud_cover']}% — {region}")
        except Exception as e:
            logger.warning(f"OpenWeatherMap hatası: {e}")

    if omm_data is None and owm_data is None:
        raise RuntimeError("Her iki hava durumu API'si de yanıt vermedi.")

    # ── Fusion ───────────────────────────────────────────────
    omm_cloud = omm_data["cloud_cover"] if omm_data else None
    owm_cloud  = owm_data["cloud_cover"] if owm_data else None

    if omm_cloud is not None and owm_cloud is not None and omm_cloud >= 0 and owm_cloud >= 0:
        # İki kaynak ortalaması — daha güvenilir ölçüm
        fused_cloud = (omm_cloud + owm_cloud) // 2
        source = "Fused (OMM + OWM)"
    elif omm_cloud is not None and omm_cloud >= 0:
        fused_cloud = omm_cloud
        source = "Open-Meteo"
    else:
        fused_cloud = owm_cloud if owm_cloud is not None and owm_cloud >= 0 else 0
        source = "OpenWeatherMap"

    # Sıcaklık, rüzgar, yağış — önce OWM (m/s→km/h zaten çevrildi), yoksa OMM
    base = owm_data if owm_data else omm_data
    temp   = float(base.get("temperature",   omm_data["temperature"]   if omm_data else 0.0))
    wind   = float(base.get("wind_speed",    omm_data["wind_speed"]    if omm_data else 0.0))
    precip = float(base.get("precipitation", omm_data["precipitation"] if omm_data else 0.0))
    vis    = int(base.get("visibility", 10000))
    desc   = owm_data.get("description", "") if owm_data else ""

    return WeatherReading(
        lat          = lat,
        lon          = lon,
        cloud_cover  = fused_cloud,
        temperature  = temp,
        wind_speed   = wind,
        precipitation= precip,
        visibility   = vis,
        source       = source,
        owm_cloud    = owm_cloud,
        omm_cloud    = omm_cloud,
        weather_desc = desc,
    )


def make_decision(region: str, weather: WeatherReading) -> SensorDecision:
    """
    Hava durumuna göre otonom uydu sensör kararı üretir.

    Algoritma:
      efektif_bulut = bulut + yağış_cezası (her mm yağış için +10%)
      ≤ 20% → İMECE (millî optik, 0.9m GSD)
      21-60% → Planet SkySat (ticari optik, 0.5m GSD)
      > 60% → Göktürk-2 (SAR, bulut/gece etkisiz)
    """
    precip_penalty = min(30, int(weather.precipitation * 10))
    eff = min(100, weather.cloud_cover + precip_penalty)

    if eff <= 20:
        satellite = "İMECE"
        status    = "OPEN"
        conf      = "HIGH"
        short     = f"Hava açık (%{eff} bulut). İMECE uydusu görevlendirildi."
        long_     = (
            f"Bulut örtüsü %{weather.cloud_cover} — optik görüntüleme için ideal koşullar. "
            f"İMECE'nin 0.9 m GSD çözünürlüğü yapı hasarını, yol kapanmalarını ve "
            f"heyelan alanlarını net olarak tespit edebilir. SAR'a geçiş gerekli değil."
        )
    elif eff <= 60:
        satellite = "Planet SkySat"
        status    = "PARTIAL"
        conf      = "MEDIUM"
        short     = f"Kısmi bulutluluk (%{eff}). Planet SkySat devreye alındı."
        long_     = (
            f"Bulut örtüsü %{weather.cloud_cover} (yağış cezası +{precip_penalty}% = %{eff} efektif). "
            f"İMECE için görüş kısıtlı. Planet SkySat'ın 0.5 m GSD ve yüksek tekrar geçiş "
            f"kapasitesi boşlukları kapatır. SAR yedekte hazır."
        )
    else:
        satellite = "Göktürk-2"
        status    = "CLOSED"
        conf      = "HIGH"
        short     = f"Bulutluluk kritik (%{eff}). Optik görüş imkânsız. Göktürk-2 SAR devrede."
        long_     = (
            f"Bulut örtüsü %{weather.cloud_cover} (yağış cezası +{precip_penalty}% = %{eff} efektif). "
            f"Tüm optik uyduların görüş kapasitesi sıfır. Sistem otonom olarak Göktürk-2 "
            f"SAR Radar'a geçiş yaptı. SAR, bulut altını ve geceyi doğrudan görüntüler. "
            f"C-band SAR sinyalleri su baskını, çökme alanları ve enkaz yığılmasını "
            f"optikten bağımsız olarak tespit eder."
        )

    return SensorDecision(
        region          = region,
        weather         = weather,
        satellite       = satellite,
        sensor_type     = SAT_PROFILES[satellite]["type"],
        status          = status,
        confidence      = conf,
        decision_text   = short,
        decision_long   = long_,
        effective_cloud = eff,
    )


def fetch_and_decide(region: str, owm_api_key: Optional[str] = None) -> SensorDecision:
    """
    fetch_weather + make_decision birleşik çağrı.

    Args:
        region     : Afet bölgesi adı (DISASTER_REGIONS anahtarı)
        owm_api_key: OpenWeatherMap API anahtarı (opsiyonel — verilirse dual-source fusion)
    """
    weather = fetch_weather(region, owm_api_key=owm_api_key)
    return make_decision(region, weather)
