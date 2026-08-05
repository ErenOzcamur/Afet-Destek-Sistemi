# ============================================================
# planet_client.py - Planet Labs Data API v1 İstemcisi
# TUA Astro Hackathon DSS
# ============================================================

"""
Planet Data API v1 entegrasyonu.

Desteklenen ürün tipleri:
  - PSScene      → PlanetScope  ~3 m/px
  - SkySatScene  → SkySat       ~0.5 m/px

Kullanım:
    client = PlanetClient(api_key="PL-...")
    scenes = client.search_scenes(
        bbox=(37.0, 36.5, 37.5, 37.0),   # (lon_min, lat_min, lon_max, lat_max)
        date_from="2024-01-17T00:00:00Z",
        date_to="2024-01-19T23:59:59Z",
        item_types=["PSScene"],
        limit=10,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

# ── Sabitler ────────────────────────────────────────────────
BASE_URL = "https://api.planet.com/data/v1"
QUICK_SEARCH_URL = f"{BASE_URL}/quick-search"

ITEM_TYPE_RESOLUTION: Dict[str, float] = {
    "PSScene": 3.0,         # PlanetScope — multispektral, ~3 m/px
    "SkySatScene": 0.5,     # SkySat — pankromatik, ~0.5 m/px
    "SkySatCollect": 0.5,
}

ITEM_TYPE_LABEL: Dict[str, str] = {
    "PSScene": "PlanetScope (3m)",
    "SkySatScene": "SkySat (0.5m)",
    "SkySatCollect": "SkySat Collect (0.5m)",
}

REQUEST_TIMEOUT = (5, 15)  # (connect, read) saniye
DEFAULT_RESOLUTION_M = 3.0
SEARCH_NAME = "astro_resq_search"
MAX_PAGES = 10
MAX_PAGE_SIZE = 250


@dataclass
class PlanetScene:
    """Tek bir Planet sahnesini temsil eder."""
    scene_id: str
    item_type: str
    acquired: datetime
    cloud_cover: float
    resolution_m: float
    thumbnail_url: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def acquired_tr(self) -> str:
        """TR formatında tarih."""
        return self.acquired.strftime("%d.%m.%Y %H:%M UTC")

    @property
    def label(self) -> str:
        return f"{ITEM_TYPE_LABEL.get(self.item_type, self.item_type)} — {self.acquired_tr}"

    def to_row(self) -> Dict[str, Any]:
        return {
            "Sahne ID": self.scene_id,
            "Tip": ITEM_TYPE_LABEL.get(self.item_type, self.item_type),
            "Tarih": self.acquired_tr,
            "Bulut %": f"{self.cloud_cover * 100:.1f}%",
            "Çözünürlük": f"{self.resolution_m} m/px",
        }


class PlanetClient:
    """
    Planet Data API v1 istemcisi.

    Kimlik doğrulama: Basic Auth (api_key = kullanıcı adı, şifre boş).
    """

    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("Planet API anahtarı boş olamaz.")
        self._api_key = api_key.strip()
        self._session = requests.Session()
        self._session.auth = (self._api_key, "")
        self._session.headers.update({"Content-Type": "application/json"})
        logger.debug("PlanetClient başlatıldı.")

    def close(self) -> None:
        """Oturumu (session) kapatır ve bağlantı kaynaklarını serbest bırakır."""
        self._session.close()

    def __enter__(self) -> "PlanetClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Public API ──────────────────────────────────────────

    def validate_key(self) -> bool:
        """API anahtarının geçerli olup olmadığını kontrol eder."""
        try:
            r = self._session.get(f"{BASE_URL}/item-types", timeout=REQUEST_TIMEOUT)
            return r.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Planet key doğrulama hatası: {e}")
            return False

    def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        date_from: str,
        date_to: str,
        item_types: Optional[List[str]] = None,
        cloud_cover_max: float = 0.20,
        limit: int = 20,
    ) -> List[PlanetScene]:
        """
        Belirtilen bounding box ve tarih aralığında Planet sahnelerini arar.

        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            date_from: ISO 8601 başlangıç tarihi, örn "2024-01-01T00:00:00Z"
            date_to: ISO 8601 bitiş tarihi
            item_types: ["PSScene"] veya ["SkySatScene"]
            cloud_cover_max: Maksimum bulut örtüsü oranı (0–1)
            limit: Maksimum sonuç sayısı

        Returns:
            PlanetScene listesi (tarih azalan sırada)
        """
        resolved_item_types = list(item_types) if item_types else ["PSScene"]
        lon_min, lat_min, lon_max, lat_max = bbox

        geojson_geometry = {
            "type": "Polygon",
            "coordinates": [[
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]],
        }

        payload = {
            "name": SEARCH_NAME,
            "item_types": resolved_item_types,
            "filter": {
                "type": "AndFilter",
                "config": [
                    {
                        "type": "GeometryFilter",
                        "field_name": "geometry",
                        "config": geojson_geometry,
                    },
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {"gte": date_from, "lte": date_to},
                    },
                    {
                        "type": "RangeFilter",
                        "field_name": "cloud_cover",
                        "config": {"lte": cloud_cover_max},
                    },
                ],
            },
        }

        scenes: List[PlanetScene] = []
        page_size = min(limit, MAX_PAGE_SIZE)
        first_url = f"{QUICK_SEARCH_URL}?_page_size={page_size}"
        next_url: Optional[str] = first_url
        page_count = 0

        try:
            while next_url and len(scenes) < limit:
                page_count += 1
                if page_count > MAX_PAGES:
                    logger.warning(
                        f"Maksimum sayfa sayısına ulaşıldı ({MAX_PAGES}); eldeki {len(scenes)} sonuç döndürülüyor."
                    )
                    break
                if next_url == first_url:
                    r = self._session.post(next_url, json=payload, timeout=REQUEST_TIMEOUT)
                else:
                    r = self._session.get(next_url, timeout=REQUEST_TIMEOUT)

                if r.status_code == 401:
                    raise PermissionError("Geçersiz Planet API anahtarı.")
                r.raise_for_status()

                data = r.json()
                for feature in data.get("features", []):
                    scene = self._parse_feature(feature)
                    if scene:
                        scenes.append(scene)
                    if len(scenes) >= limit:
                        break

                next_url = data.get("_links", {}).get("_next")

        except PermissionError:
            raise
        except requests.RequestException as e:
            logger.error(f"Planet arama hatası: {e}")
            raise ConnectionError(f"Planet API bağlantı hatası: {e}") from e

        logger.info(f"Planet arama: {len(scenes)} sahne bulundu ({', '.join(resolved_item_types)})")
        return scenes

    def get_thumbnail_url(self, scene: PlanetScene) -> str:
        """Sahnenin thumbnail URL'ini döndürür (auth header gerektirir).

        Not: Basit erişimcidir; yeni kod doğrudan `scene.thumbnail_url` kullanmalı.
        Geriye dönük uyumluluk (public API) için korunmaktadır.
        """
        return scene.thumbnail_url

    # ── Internal ────────────────────────────────────────────

    def _parse_feature(self, feature: Dict) -> Optional[PlanetScene]:
        """GeoJSON feature'ı PlanetScene'e dönüştürür."""
        try:
            props = feature.get("properties", {})
            item_type = props.get("item_type") or feature.get("id", "").split("_")[0]

            # Tarih parse
            acquired_str = props.get("acquired", "")
            try:
                acquired = datetime.fromisoformat(
                    acquired_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                logger.warning(f"'acquired' parse edilemedi: {acquired_str!r}")
                acquired = datetime.now(timezone.utc)

            thumbnail = (
                feature.get("_links", {}).get("thumbnail")
                or feature.get("assets", {}).get("thumbnail", {}).get("location", "")
            )

            raw_cc = props.get("cloud_cover")
            cloud_cover = float(raw_cc) if raw_cc is not None else 0.0

            return PlanetScene(
                scene_id=feature.get("id", ""),
                item_type=item_type,
                acquired=acquired,
                cloud_cover=cloud_cover,
                resolution_m=ITEM_TYPE_RESOLUTION.get(item_type, DEFAULT_RESOLUTION_M),
                thumbnail_url=thumbnail,
                geometry=feature.get("geometry", {}),
                properties=props,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Sahne parse hatası (id={feature.get('id')}): {e}")
            return None
