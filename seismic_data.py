# ============================================================
# seismic_data.py - Canlı Sismik Veri Modülü
# TUA Astro Hackathon 2026
# ============================================================
"""
Üç kaynaktan canlı deprem verisi çeker ve ortak formata normalize eder.

Kaynaklar:
  AFAD     : https://deprem.afad.gov.tr/apiv2/event/filter
  Kandilli : https://api.orhanaydogdu.com.tr/deprem/kandilli/live
  USGS     : https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson

Normalize Çıktı:
  Her kaynak → EarthquakeEvent listesi → birleştirilmiş, sıralanmış feed

Not: AFAD ve Kandilli Türkiye odaklı, USGS global kapsama sahiptir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from loguru import logger

# ────────────────────────────────────────────────────────────
# SABİTLER
# ────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 10  # saniye
REQUEST_HEADERS = {
    "User-Agent": "ASTRO-RESQ/2.0 TUA-Hackathon (contact@astro-resq.tr)",
    "Accept": "application/json",
}

MAG_COLOR = {
    "low":      "#2ecc71",   # < 3.0  → yeşil
    "moderate": "#f39c12",   # 3.0-4.0 → turuncu
    "strong":   "#e67e22",   # 4.0-5.0 → koyu turuncu
    "major":    "#e74c3c",   # 5.0-6.0 → kırmızı
    "great":    "#8e44ad",   # ≥ 6.0   → mor
}

SOURCE_COLOR = {
    "AFAD":     "#e63946",
    "Kandilli": "#f77f00",
    "USGS":     "#0077b6",
}

TZ_TR = ZoneInfo("Europe/Istanbul")

DEFAULT_SOURCES: Tuple[str, ...] = ("AFAD", "Kandilli", "USGS")
DEFAULT_FETCH_LIMIT = 500

MIN_MARKER_RADIUS_PX = 4
MARKER_RADIUS_EXPONENT = 2.0

# (üst sınır, seviye) — sıralı; üstüne çıkanlar "great"
MAG_LEVEL_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (3.0, "low"),
    (4.0, "moderate"),
    (5.0, "strong"),
    (6.0, "major"),
)


# ────────────────────────────────────────────────────────────
# VERİ MODELİ
# ────────────────────────────────────────────────────────────

@dataclass
class EarthquakeEvent:
    """Normalize edilmiş deprem olayı."""

    event_id:     str
    source:       str            # "AFAD" | "Kandilli" | "USGS"
    magnitude:    float
    depth_km:     float
    latitude:     float
    longitude:    float
    location:     str
    datetime_utc: datetime
    mag_type:     str  = "ML"   # "ML" | "Mw" | "mb" vb.
    province:     str  = ""
    district:     str  = ""
    url:          str  = ""

    # ── Türetilen özellikler ──────────────────────────────

    @property
    def mag_level(self) -> str:
        for threshold, level in MAG_LEVEL_THRESHOLDS:
            if self.magnitude < threshold:
                return level
        return "great"

    @property
    def color(self) -> str:
        return MAG_COLOR[self.mag_level]

    @property
    def radius(self) -> float:
        """Harita marker yarıçapı (px)."""
        return max(MIN_MARKER_RADIUS_PX, self.magnitude ** MARKER_RADIUS_EXPONENT)

    @property
    def datetime_tr(self) -> str:
        """Türkiye saatine çevrilmiş okunabilir zaman."""
        try:
            local = self.datetime_utc.astimezone(TZ_TR)
            return local.strftime("%d.%m.%Y %H:%M:%S (TR)")
        except (ValueError, OSError) as e:
            logger.debug(f"datetime_tr dönüşüm hatası: {e}")
            return self.datetime_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    @property
    def age_minutes(self) -> float:
        """Kaç dakika önce gerçekleşti."""
        now = datetime.now(timezone.utc)
        return (now - self.datetime_utc).total_seconds() / 60

    def to_row(self) -> Dict:
        """Streamlit DataFrame satırı."""
        return {
            "Kaynak":    self.source,
            "Büyüklük":  f"{self.magnitude:.1f} {self.mag_type}",
            "Derinlik":  f"{self.depth_km:.0f} km",
            "Konum":     self.location,
            "Tarih/Saat": self.datetime_tr,
            "Önce":      f"{self.age_minutes:.0f} dk",
        }


# ────────────────────────────────────────────────────────────
# FETCHER
# ────────────────────────────────────────────────────────────

class SeismicFetcher:
    """
    Üç farklı kaynaktan canlı deprem verisi çeker ve birleştirir.

    Kullanım:
        fetcher = SeismicFetcher()
        events  = fetcher.fetch_all(hours=24, min_mag=1.0)
        stats   = fetcher.last_fetch_stats
    """

    AFAD_URL     = "https://deprem.afad.gov.tr/apiv2/event/filter"
    KANDILLI_URL = "https://api.orhanaydogdu.com.tr/deprem/kandilli/live"
    USGS_URL     = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(self):
        self.last_fetch_stats: Dict = {}
        self._session = requests.Session()
        self._session.headers.update(REQUEST_HEADERS)

    def close(self) -> None:
        """HTTP oturumunu kapatır (socket/bellek sızıntısını önler)."""
        try:
            self._session.close()
        except Exception as e:
            logger.debug(f"Session kapatma hatası: {e}")

    def __enter__(self) -> "SeismicFetcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        try:
            self._session.close()
        except Exception:
            pass

    # ════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════

    def fetch_all(
        self,
        hours: int = 24,
        min_mag: float = 1.0,
        sources: Optional[List[str]] = None,
    ) -> List[EarthquakeEvent]:
        """
        Tüm kaynaklardan deprem verisini çekip birleştirir.

        Args:
            hours:    Kaç saatlik veri (1-168)
            min_mag:  Minimum büyüklük filtresi
            sources:  ["AFAD", "Kandilli", "USGS"] — None=hepsi

        Returns:
            Yeniden eskiye sıralı EarthquakeEvent listesi
        """
        sources = list(sources or DEFAULT_SOURCES)
        t0 = time.time()
        all_events: List[EarthquakeEvent] = []
        source_stats: Dict[str, Dict] = {}

        for name in DEFAULT_SOURCES:
            if name not in sources:
                continue
            evts, ok = self.fetch_source(name, hours=hours, min_mag=min_mag)
            all_events.extend(evts)
            source_stats[name] = {"count": len(evts), "ok": ok}

        # Sırala: yeniden eskiye
        all_events.sort(key=lambda e: e.datetime_utc, reverse=True)

        # Stats'i tek seferde, yeni bir dict olarak oluştur (immutability)
        stats: Dict = {
            "sources": source_stats,
            "total": len(all_events),
            "elapsed_s": round(time.time() - t0, 2),
        }
        self.last_fetch_stats = stats

        logger.info(
            f"Sismik veri: {stats['total']} deprem | "
            + " | ".join(
                f"{s}:{d['count']}" for s, d in stats["sources"].items()
            )
            + f" | {stats['elapsed_s']}s"
        )
        return all_events

    def fetch_source(
        self,
        source: str,
        hours: int = 24,
        min_mag: float = 1.0,
    ) -> Tuple[List[EarthquakeEvent], bool]:
        """Tek bir kaynaktan veri çeker."""
        dispatch = {
            "AFAD":     lambda: self._fetch_afad(hours=hours, min_mag=min_mag),
            "Kandilli": lambda: self._fetch_kandilli(min_mag=min_mag),
            "USGS":     lambda: self._fetch_usgs(hours=hours, min_mag=min_mag),
        }
        fetcher = dispatch.get(source)
        if fetcher is None:
            logger.warning(f"Bilinmeyen kaynak: {source!r}")
            return [], False
        return fetcher()

    def _parse_and_filter(
        self,
        items: List,
        parser,
        min_mag: float,
    ) -> List[EarthquakeEvent]:
        """Ham item listesini parse edip min_mag filtresi uygular."""
        return [
            evt
            for evt in (parser(item) for item in items)
            if evt is not None and evt.magnitude >= min_mag
        ]

    # ════════════════════════════════════════════════════════
    # AFAD
    # ════════════════════════════════════════════════════════

    def _fetch_afad(
        self,
        hours: int = 24,
        min_mag: float = 1.0,
        limit: int = DEFAULT_FETCH_LIMIT,
    ) -> Tuple[List[EarthquakeEvent], bool]:
        """
        AFAD Deprem API.
        Endpoint: https://deprem.afad.gov.tr/apiv2/event/filter
        Parametreler: start, end, minmag, limit, orderby
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        params = {
            "start":    start.strftime("%Y-%m-%d %H:%M:%S"),
            "end":      now.strftime("%Y-%m-%d %H:%M:%S"),
            "minmag":   min_mag,
            "limit":    limit,
            "orderby":  "timedesc",
        }

        try:
            resp = self._session.get(
                self.AFAD_URL, params=params,
                timeout=REQUEST_TIMEOUT, allow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                logger.warning(f"AFAD: Beklenmeyen yanıt formatı: {type(data)}")
                return [], False

            events = self._parse_and_filter(data, self._parse_afad_item, min_mag)

            logger.debug(f"AFAD: {len(events)} deprem alındı.")
            return events, True

        except requests.exceptions.Timeout:
            logger.warning("AFAD API: Zaman aşımı.")
            return [], False
        except requests.exceptions.ConnectionError:
            logger.warning("AFAD API: Bağlantı hatası.")
            return [], False
        except requests.exceptions.RequestException as e:
            logger.error(f"AFAD API HTTP hatası: {e}")
            return [], False
        except ValueError as e:
            logger.error(f"AFAD API JSON parse hatası: {e}")
            return [], False
        except Exception:
            logger.exception("AFAD API beklenmeyen hata")
            return [], False

    def _parse_afad_item(self, item: Dict) -> Optional[EarthquakeEvent]:
        """AFAD API yanıtındaki tek bir eventi parse eder."""
        try:
            lat = float(item.get("latitude", 0))
            lon = float(item.get("longitude", 0))
            mag = float(item.get("magnitude", 0))
            depth = float(item.get("depth", 0))

            # Tarih parse
            date_str = item.get("date", "")
            try:
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                logger.debug(f"AFAD tarih parse edilemedi: {date_str!r}")
                dt = datetime.now(timezone.utc)

            location = item.get("location", "")
            province = item.get("province", "")
            district = item.get("district", "")
            if not location:
                location = f"{district}, {province}".strip(", ")

            return EarthquakeEvent(
                event_id     = f"AFAD-{item.get('eventID', '')}",
                source       = "AFAD",
                magnitude    = mag,
                depth_km     = depth,
                latitude     = lat,
                longitude    = lon,
                location     = location,
                datetime_utc = dt,
                mag_type     = item.get("type", "ML"),
                province     = province,
                district     = district,
            )
        except Exception as e:
            logger.debug(f"AFAD item parse hatası: {e}")
            return None

    # ════════════════════════════════════════════════════════
    # KANDİLLİ
    # ════════════════════════════════════════════════════════

    def _fetch_kandilli(
        self,
        min_mag: float = 1.0,
    ) -> Tuple[List[EarthquakeEvent], bool]:
        """
        Orhan Aydoğdu — Kandilli Rasathanesi feed.
        Endpoint: https://api.orhanaydogdu.com.tr/deprem/kandilli/live
        Canlı feed, son ~50 deprem.
        """
        try:
            resp = self._session.get(
                self.KANDILLI_URL, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, dict):
                logger.warning("Kandilli: Beklenmeyen yanıt formatı")
                return [], False

            if not data.get("status"):
                logger.warning("Kandilli: status=False")
                return [], False

            result = data.get("result", [])
            events = self._parse_and_filter(
                result, self._parse_kandilli_item, min_mag
            )

            logger.debug(f"Kandilli: {len(events)} deprem alındı.")
            return events, True

        except requests.exceptions.Timeout:
            logger.warning("Kandilli API: Zaman aşımı.")
            return [], False
        except requests.exceptions.ConnectionError:
            logger.warning("Kandilli API: Bağlantı hatası.")
            return [], False
        except requests.exceptions.RequestException as e:
            logger.error(f"Kandilli API HTTP hatası: {e}")
            return [], False
        except ValueError as e:
            logger.error(f"Kandilli API JSON parse hatası: {e}")
            return [], False
        except Exception:
            logger.exception("Kandilli API beklenmeyen hata")
            return [], False

    def _parse_kandilli_item(self, item: Dict) -> Optional[EarthquakeEvent]:
        """Kandilli API yanıtındaki tek bir eventi parse eder."""
        try:
            coords = item.get("geojson", {}).get("coordinates", [0, 0])
            lon = float(coords[0])
            lat = float(coords[1])
            mag = float(item.get("mag", 0))
            depth = float(item.get("depth", 0))

            # Tarih parse (format: "2026-03-28 11:57:06")
            date_str = item.get("date_time", "")
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                # Kandilli verisi Türkiye saatinde → UTC'ye çevir
                dt = dt.replace(tzinfo=TZ_TR).astimezone(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)

            loc_props = item.get("location_properties", {})
            epi = loc_props.get("epiCenter", {})
            province = epi.get("name", "")
            title = item.get("title", province)

            return EarthquakeEvent(
                event_id     = f"Kandilli-{item.get('earthquake_id', '')}",
                source       = "Kandilli",
                magnitude    = mag,
                depth_km     = depth,
                latitude     = lat,
                longitude    = lon,
                location     = title,
                datetime_utc = dt,
                mag_type     = "ML",
                province     = province,
            )
        except Exception as e:
            logger.debug(f"Kandilli item parse hatası: {e}")
            return None

    # ════════════════════════════════════════════════════════
    # USGS
    # ════════════════════════════════════════════════════════

    def _fetch_usgs(
        self,
        hours: int = 24,
        min_mag: float = 1.0,
        limit: int = 500,
        region: Optional[Tuple] = None,
    ) -> Tuple[List[EarthquakeEvent], bool]:
        """
        USGS FDSN Earthquake API.
        Endpoint: https://earthquake.usgs.gov/fdsnws/event/1/query
        Global kapsam; gerçek zamanlı ~1 dk gecikme.

        Args:
            region: (minlat, maxlat, minlon, maxlon) — None=global
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        params = {
            "format":     "geojson",
            "starttime":  start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime":    now.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": min_mag,
            "limit":      limit,
            "orderby":    "time",
        }
        if region:
            params["minlatitude"]  = region[0]
            params["maxlatitude"]  = region[1]
            params["minlongitude"] = region[2]
            params["maxlongitude"] = region[3]

        try:
            resp = self._session.get(
                self.USGS_URL, params=params, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            events = []
            for feat in features:
                evt = self._parse_usgs_feature(feat)
                if evt:
                    events.append(evt)

            logger.debug(f"USGS: {len(events)} deprem alındı.")
            return events, True

        except requests.exceptions.Timeout:
            logger.warning("USGS API: Zaman aşımı.")
            return [], False
        except Exception as e:
            logger.error(f"USGS API hatası: {e}")
            return [], False

    def _parse_usgs_feature(self, feat: Dict) -> Optional[EarthquakeEvent]:
        """USGS GeoJSON feature'ını parse eder."""
        try:
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])

            lon   = float(coords[0])
            lat   = float(coords[1])
            depth = float(coords[2]) if len(coords) > 2 else 0.0
            mag   = float(props.get("mag") or 0)

            # Zaman: epoch ms → UTC datetime
            ts_ms = props.get("time", 0)
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

            place    = props.get("place", "")
            mag_type = props.get("magType", "Mw")
            url      = props.get("url", "")
            evt_id   = feat.get("id", "")

            return EarthquakeEvent(
                event_id     = f"USGS-{evt_id}",
                source       = "USGS",
                magnitude    = mag,
                depth_km     = depth,
                latitude     = lat,
                longitude    = lon,
                location     = place,
                datetime_utc = dt,
                mag_type     = mag_type,
                url          = url,
            )
        except Exception as e:
            logger.debug(f"USGS feature parse hatası: {e}")
            return None

    # ════════════════════════════════════════════════════════
    # YARDIMCI
    # ════════════════════════════════════════════════════════

    @staticmethod
    def summary_stats(events: List[EarthquakeEvent]) -> Dict:
        """Deprem listesinden özet istatistikler üretir."""
        if not events:
            return {
                "total": 0, "max_mag": 0, "avg_mag": 0,
                "by_level": {}, "by_source": {},
            }

        mags = [e.magnitude for e in events]
        by_level:  Dict[str, int] = {}
        by_source: Dict[str, int] = {}

        for e in events:
            by_level[e.mag_level]  = by_level.get(e.mag_level, 0)  + 1
            by_source[e.source]    = by_source.get(e.source, 0)    + 1

        return {
            "total":     len(events),
            "max_mag":   max(mags),
            "avg_mag":   round(sum(mags) / len(mags), 2),
            "by_level":  by_level,
            "by_source": by_source,
            "latest":    events[0].datetime_tr if events else "—",
        }
