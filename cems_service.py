# ============================================================
# cems_service.py - Copernicus EMS Entegrasyon Modülü
# TUA Astro Hackathon DSS — OGC WMS/WFS + Doğrulama Motoru
# ============================================================

"""
Copernicus Emergency Management Service (CEMS) entegrasyonu.

Servisler:
  - WMS : Görsel hasar katmanları (GeoServer tabanlı, OGC uyumlu)
  - WFS : Vektör hasar verileri — GeoJSON Polygon/LineString

EMSR kodu örn: EMSR648 → 2023 Türkiye-Suriye depremi aktivasyonu
Aktivasyon listesi: https://emergency.copernicus.eu/mapping/list-of-activations-rapid

Doğrulama Motoru:
  - Yerli uydu (İMECE/GÖKTÜRK) → binary hasar maskesi
  - CEMS resmi vektörü → referans Polygon listesi
  - IoU = Intersection / Union (piksel veya alan bazlı)
  - Precision / Recall / F1 hesabı
  - False Positive / False Negative harita katmanları

OGC standart endpoint'leri:
  WMS: https://emergency.copernicus.eu/mapping/wms/EMSR648
  WFS: https://emergency.copernicus.eu/mapping/wfs
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from loguru import logger


# ── Sabitler ────────────────────────────────────────────────

CEMS_WMS_BASE = "https://emergency.copernicus.eu/mapping/wms"
CEMS_WFS_BASE = "https://emergency.copernicus.eu/mapping/wfs"
CEMS_API_BASE = "https://emergency.copernicus.eu/mapping/activations"

REQUEST_TIMEOUT = 20

# Harita çizim sabitleri
MARKER_RADIUS_PX = 6
POPUP_MAX_WIDTH_PX = 220
POLYGON_WEIGHT = 1.5
LINE_WEIGHT = 3
# Yarı saydam overlay alfa değeri (0-255)
OVERLAY_ALPHA = 160

# Coğrafi yaklaşık dönüşüm sabitleri
METERS_PER_DEGREE = 111_320   # derece → metre (ekvatorda yaklaşık)
ROAD_WIDTH_M = 6              # varsayılan yol genişliği (m)

# CEMS katman isimleri (EMSR kodu ile birleştirilir)
LAYER_TYPES = {
    "overview":          "Overview Extent",
    "observed_event":    "Observed Event",
    "road_damage":       "Damaged Roads",
    "building_damage":   "Damaged Buildings",
    "temporary_shelter": "Temporary Shelters",
    "flooded_area":      "Flooded Area",
}

# Hasar derecesi renkleri (haritada popup için)
DAMAGE_GRADE_COLOR = {
    "Destroyed":          "#e63946",
    "Highly Damaged":     "#f77f00",
    "Moderately Damaged": "#fcbf49",
    "Negligible to slight damage": "#06d6a0",
    "Unknown":            "#adb5bd",
}


# ── Dataclass'lar ────────────────────────────────────────────

@dataclass
class CEMSLayer:
    """Tek bir WMS katmanını tanımlar."""
    layer_name: str          # Sunucudaki gerçek layer adı
    label:      str          # UI etiketi
    wms_url:    str          # Tam WMS endpoint
    emsr_code:  str


@dataclass
class CEMSFeature:
    """Tek bir vektör özellik (WFS'ten gelen)."""
    feature_id:   str
    geometry:     Dict        # GeoJSON geometry
    damage_grade: str
    obj_type:     str         # building / road / area
    area_m2:      float
    centroid:     Tuple[float, float]   # (lat, lon)
    properties:   Dict = field(default_factory=dict)

    @property
    def color(self) -> str:
        return DAMAGE_GRADE_COLOR.get(self.damage_grade, "#adb5bd")

    def popup_html(self) -> str:
        return (
            f"<b>CEMS — {self.obj_type.upper()}</b><br>"
            f"Hasar: <b style='color:{self.color}'>{self.damage_grade}</b><br>"
            f"Alan: {self.area_m2:.1f} m²<br>"
            f"ID: {self.feature_id}"
        )


@dataclass
class ValidationReport:
    """Doğrulama motoru çıktısı."""
    emsr_code:       str
    satellite_source: str
    iou:             float
    precision:       float
    recall:          float
    f1_score:        float
    accuracy:        float
    true_positives:  int
    false_positives: int
    false_negatives: int
    true_negatives:  int
    total_pixels:    int
    overlap_area_m2: float
    union_area_m2:   float
    fp_mask:         Optional[np.ndarray] = None   # False Positive harita maskesi
    fn_mask:         Optional[np.ndarray] = None   # False Negative harita maskesi

    def summary(self) -> Dict:
        return {
            "emsr_code":       self.emsr_code,
            "satellite_source": self.satellite_source,
            "iou":             round(self.iou, 4),
            "precision":       round(self.precision, 4),
            "recall":          round(self.recall, 4),
            "f1_score":        round(self.f1_score, 4),
            "accuracy":        round(self.accuracy, 4),
            "true_positives":  self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.summary(), ensure_ascii=False, indent=indent)


# ── Ana Servis Sınıfı ────────────────────────────────────────

class CEMSService:
    """
    Copernicus EMS WMS/WFS istemcisi + Doğrulama Motoru.

    Kullanım:
        svc = CEMSService()
        layers  = svc.get_wms_layers("EMSR648")
        features = svc.fetch_wfs_features("EMSR648", layer_type="building_damage")
        report  = svc.validate_against_cems(
            binary_mask, features, resolution_m=0.5, geo_transform=tf
        )
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ASTRO-RESQ/2.0 TUA-Hackathon"})
        self._feature_cache: Dict[str, List[CEMSFeature]] = {}

    # ── WMS Katman Yönetimi ──────────────────────────────────

    def get_wms_layers(self, emsr_code: str) -> List[CEMSLayer]:
        """
        EMSR koduna göre mevcut WMS katmanlarını listeler.

        Args:
            emsr_code: "EMSR648" gibi aktivasyon kodu

        Returns:
            CEMSLayer listesi (UI'da seçim için)
        """
        code = emsr_code.strip().upper()
        wms_url = f"{CEMS_WMS_BASE}/{code}"

        layers = []
        try:
            # GetCapabilities ile mevcut katmanları sorgula
            r = self._session.get(
                wms_url,
                params={"SERVICE": "WMS", "VERSION": "1.3.0",
                        "REQUEST": "GetCapabilities"},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            if r.status_code == 200:
                # Katman listesi statik üretilir (GetCapabilities yalnızca
                # erişilebilirlik kontrolü — yanıt gövdesi tutulmaz)
                layers = self._static_layers(code, wms_url)
                logger.info(f"CEMS WMS: {len(layers)} katman — {code}")
            else:
                # Sunucu erişilemez → sabit bilinen katmanları döndür
                layers = self._static_layers(code, wms_url)
                logger.warning(f"CEMS WMS GetCapabilities {r.status_code} — statik liste")
        except requests.RequestException as exc:
            layers = self._static_layers(code, wms_url)
            logger.warning(f"CEMS WMS bağlantı hatası: {exc} — statik liste")

        return layers

    def build_wms_tile_url(
        self, layer: CEMSLayer
    ) -> str:
        """
        leafmap / Folium TileLayer için WMS tile URL şablonu.
        {z}/{x}/{y} yerine Folium'un WMS parametrelerini kullanır.

        Not: layer.wms_url üzerinde ince sarmalayıcıdır (public API
        sözleşmesi gereği korunur).

        Returns:
            Folium WMS katman eklemek için tam base URL
        """
        return layer.wms_url

    def add_wms_to_folium(
        self,
        folium_map,
        layer: CEMSLayer,
        opacity: float = 0.7,
        overlay: bool = True,
    ):
        """
        Folium haritasına CEMS WMS katmanını overlay olarak ekler.

        Args:
            folium_map: Folium Map nesnesi
            layer     : CEMSLayer (get_wms_layers() çıktısı)
            opacity   : Katman saydamlığı (0–1)
            overlay   : True = overlay katman, False = base katman

        OGC WMS standardı — BBOX ve FORMAT parametreleri otomatik ayarlanır.
        """
        try:
            import folium
            folium.WmsTileLayer(
                url      = layer.wms_url,
                name     = f"CEMS — {layer.label}",
                layers   = layer.layer_name,
                fmt      = "image/png",
                transparent=True,
                version  = "1.3.0",
                attr     = "© Copernicus EMS",
                overlay  = overlay,
                control  = True,
                opacity  = opacity,
            ).add_to(folium_map)
            logger.debug(f"WMS katmanı eklendi: {layer.layer_name}")
        except ImportError:
            logger.error("folium kurulu değil — WMS katmanı eklenemiyor")
        except Exception as exc:
            logger.error(f"WMS katman ekleme hatası ({type(exc).__name__}): {exc}")

    def add_wms_to_leafmap(
        self,
        lmap,
        layer: CEMSLayer,
        opacity: float = 0.7,
    ):
        """
        leafmap haritasına CEMS WMS katmanını ekler.
        Split-panel'da üçüncü katman olarak kullanılabilir.
        """
        try:
            lmap.add_wms_layer(
                url     = layer.wms_url,
                layers  = layer.layer_name,
                name    = f"CEMS — {layer.label}",
                format  = "image/png",
                transparent=True,
                opacity = opacity,
                shown   = True,
            )
            logger.debug(f"leafmap WMS katmanı eklendi: {layer.layer_name}")
        except Exception as exc:
            logger.error(f"leafmap WMS hatası: {exc}")

    # ── WFS Vektör Çekimi ────────────────────────────────────

    # Copernicus CEMS WFS katman adı kalıpları (layer_type → anahtar kelimeler)
    _WFS_KEYWORDS: Dict[str, List[str]] = {
        "road_damage":       ["road", "roads", "transportation", "transport"],
        "building_damage":   ["building", "buildings", "situationmap", "delineation"],
        "flooded_area":      ["flood", "flooded", "observedevent", "hydrography"],
        "observed_event":    ["observedevent", "observed", "event"],
        "overview":          ["overview", "aoi", "extent"],
    }

    def _discover_wfs_typenames(self, emsr_code: str) -> List[str]:
        """
        WFS GetCapabilities ile EMSR koduna ait TypeName listesini keşfeder.

        Copernicus CEMS katman isimleri çalışma zamanında değişebilir
        (EMSR648_AOI01_DEL06_AffectedRoads_l gibi). Bu metod gerçek
        isimleri doğrudan sunucudan alır.

        Returns:
            Eşleşen TypeName listesi — boş ise bağlantı başarısız
        """
        try:
            r = self._session.get(
                CEMS_WFS_BASE,
                params={"SERVICE": "WFS", "VERSION": "2.0.0",
                        "REQUEST": "GetCapabilities"},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                logger.warning(f"WFS GetCapabilities HTTP {r.status_code}")
                return []
            import re
            # XML'den Name taglerini çıkar (<Name>EMSR648_...</Name>)
            found = re.findall(
                rf'<(?:wfs:)?Name>({re.escape(emsr_code)}_[^<]+)</(?:wfs:)?Name>',
                r.text, re.IGNORECASE,
            )
            logger.info(f"WFS GetCapabilities: {len(found)} katman bulundu — {emsr_code}")
            return found
        except Exception as exc:
            logger.warning(f"WFS GetCapabilities hatası: {exc}")
            return []

    def _pick_typename(self, typenames: List[str], layer_type: str) -> Optional[str]:
        """
        Keşfedilen TypeName listesinden layer_type'a en uygun olanı seçer.
        Eşleşme: anahtar kelime içeren ilk TypeName.
        """
        keywords = self._WFS_KEYWORDS.get(layer_type, [])
        for kw in keywords:
            for tn in typenames:
                if kw.lower() in tn.lower():
                    return tn
        # Eşleşme bulunamazsa — tüm listeyi çek (ilk kayıt)
        return typenames[0] if typenames else None

    def fetch_wfs_features(
        self,
        emsr_code:  str,
        layer_type: str = "building_damage",
        bbox:       Optional[Tuple[float, float, float, float]] = None,
        max_count:  int = 500,
    ) -> List[CEMSFeature]:
        """
        CEMS WFS'ten vektör özellik çeker.

        Akış:
          1. Önbellek kontrolü
          2. GetCapabilities → gerçek TypeName keşfi (EMSR648_AOI01_..._l gibi)
          3. GetFeature → GeoJSON
          4. Parse → CEMSFeature listesi

        OGC WFS 2.0.0 standardı — OUTPUTFORMAT=application/json
        """
        code    = emsr_code.strip().upper()
        cache_k = f"{code}_{layer_type}"

        if cache_k in self._feature_cache:
            logger.debug(f"WFS cache hit: {cache_k}")
            return self._feature_cache[cache_k]

        features: List[CEMSFeature] = []

        # ── Adım 1: Gerçek TypeName'i keşfet ─────────────────────────
        typenames = self._discover_wfs_typenames(code)
        typename  = self._pick_typename(typenames, layer_type)

        if not typename:
            # Fallback: eski format dene (bazen hâlâ çalışır)
            typename = f"{code}_{layer_type}"
            logger.warning(
                f"WFS GetCapabilities'te '{code}' bulunamadı — "
                f"fallback typename: {typename}"
            )

        # ── Adım 2: GetFeature ────────────────────────────────────────
        params: Dict = {
            "SERVICE":      "WFS",
            "VERSION":      "2.0.0",
            "REQUEST":      "GetFeature",
            "TYPENAMES":    typename,
            "OUTPUTFORMAT": "application/json",
            "COUNT":        max_count,
            "SRSNAME":      "EPSG:4326",
        }
        if bbox:
            lat_min, lon_min, lat_max, lon_max = bbox
            params["BBOX"] = f"{lat_min},{lon_min},{lat_max},{lon_max},EPSG:4326"

        try:
            r = self._session.get(
                CEMS_WFS_BASE, params=params,
                timeout=REQUEST_TIMEOUT, allow_redirects=True,
            )
            if r.status_code == 200:
                try:
                    gj = r.json()
                    features = self._parse_geojson(gj, layer_type)
                    logger.info(
                        f"WFS: {len(features)} özellik çekildi — "
                        f"{code} / {typename}"
                    )
                except (json.JSONDecodeError, KeyError) as je:
                    logger.warning(f"WFS GeoJSON parse hatası: {je}")
            elif r.status_code == 404:
                logger.warning(
                    f"WFS 404 — '{typename}' katmanı bu aktivasyonda mevcut değil. "
                    f"Mevcut katmanlar: {typenames[:5]}"
                )
            else:
                logger.warning(f"WFS HTTP {r.status_code} — {typename}")
        except requests.RequestException as exc:
            logger.error(f"WFS bağlantı hatası: {exc}")

        self._feature_cache[cache_k] = features
        return features

    def add_features_to_folium(
        self,
        folium_map,
        features: List[CEMSFeature],
        group_name: str = "CEMS Hasar Verileri",
    ):
        """
        WFS özelliklerini Folium haritasına tıklanabilir Popup ile ekler.

        Her özellik → CircleMarker (bina) veya PolyLine (yol)
        """
        try:
            import folium
            fg = folium.FeatureGroup(name=group_name)

            for feat in features:
                geom = feat.geometry
                popup = folium.Popup(feat.popup_html(), max_width=220)

                if geom["type"] == "Point":
                    coords = geom["coordinates"]
                    folium.CircleMarker(
                        location=(coords[1], coords[0]),
                        radius=6,
                        color=feat.color, fill=True,
                        fill_color=feat.color, fill_opacity=0.75,
                        popup=popup,
                        tooltip=feat.damage_grade,
                    ).add_to(fg)

                elif geom["type"] in ("Polygon", "MultiPolygon"):
                    folium.GeoJson(
                        {"type": "Feature", "geometry": geom, "properties": {}},
                        style_function=lambda _, c=feat.color: {
                            "fillColor": c, "color": c,
                            "weight": 1.5, "fillOpacity": 0.45,
                        },
                        popup=popup,
                        tooltip=feat.damage_grade,
                    ).add_to(fg)

                elif geom["type"] in ("LineString", "MultiLineString"):
                    folium.GeoJson(
                        {"type": "Feature", "geometry": geom, "properties": {}},
                        style_function=lambda _, c=feat.color: {
                            "color": c, "weight": 3, "opacity": 0.8,
                        },
                        popup=popup,
                        tooltip=feat.damage_grade,
                    ).add_to(fg)

            fg.add_to(folium_map)
            logger.debug(f"Folium: {len(features)} CEMS özellik eklendi")
        except Exception as exc:
            logger.error(f"Folium CEMS özellik ekleme hatası: {exc}")

    # ── Doğrulama Motoru ─────────────────────────────────────

    def validate_against_cems(
        self,
        our_binary_mask:  np.ndarray,
        cems_features:    List[CEMSFeature],
        resolution_m:     float,
        geo_transform=None,
        image_bounds:     Optional[Tuple[float, float, float, float]] = None,
        satellite_source: str = "İMECE",
        emsr_code:        str = "EMSR???",
    ) -> ValidationReport:
        """
        Yerli uydu hasar maskesini CEMS resmi verileriyle karşılaştırır.

        Formül:
            IoU = |A ∩ B| / |A ∪ B|
            Precision = TP / (TP + FP)
            Recall    = TP / (TP + FN)
            F1        = 2 × (P × R) / (P + R)
            Accuracy  = (TP + TN) / N

        Args:
            our_binary_mask : (H, W) uint8 — bizim analiz çıktımız
            cems_features   : WFS'ten gelen resmi vektörler
            resolution_m    : m/piksel (IoU m² hesabı için)
            geo_transform   : rasterio Affine (CEMS vektör → piksel)
            image_bounds    : (lat_min, lon_min, lat_max, lon_max)
            satellite_source: "İMECE" / "GÖKTÜRK-1" / "Planet-SkySat"
            emsr_code       : Referans aktivasyon kodu

        Returns:
            ValidationReport
        """
        our_mask = (our_binary_mask > 0).astype(np.uint8)
        h, w     = our_mask.shape

        # CEMS vektörlerini referans maskesine dönüştür
        cems_mask = self._features_to_mask(
            cems_features, h, w, geo_transform, image_bounds
        )

        # ── Piksel bazlı metrikler ─────────────────────────
        tp = int(np.sum((our_mask == 1) & (cems_mask == 1)))
        fp = int(np.sum((our_mask == 1) & (cems_mask == 0)))
        fn = int(np.sum((our_mask == 0) & (cems_mask == 1)))
        tn = int(np.sum((our_mask == 0) & (cems_mask == 0)))
        n  = h * w

        px_area = resolution_m ** 2   # m²/piksel

        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp)     if (tp + fp) > 0       else 0.0
        recall    = tp / (tp + fn)     if (tp + fn) > 0       else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / n if n > 0 else 0.0

        # False Positive / Negative harita maskeleri
        fp_mask = ((our_mask == 1) & (cems_mask == 0)).astype(np.uint8) * 255
        fn_mask = ((our_mask == 0) & (cems_mask == 1)).astype(np.uint8) * 255

        logger.info(
            f"Doğrulama [{satellite_source} vs {emsr_code}]: "
            f"IoU={iou:.3f}, F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}"
        )

        return ValidationReport(
            emsr_code        = emsr_code,
            satellite_source = satellite_source,
            iou              = iou,
            precision        = precision,
            recall           = recall,
            f1_score         = f1,
            accuracy         = accuracy,
            true_positives   = tp,
            false_positives  = fp,
            false_negatives  = fn,
            true_negatives   = tn,
            total_pixels     = n,
            overlap_area_m2  = tp * px_area,
            union_area_m2    = (tp + fp + fn) * px_area,
            fp_mask          = fp_mask,
            fn_mask          = fn_mask,
        )

    def add_validation_to_folium(
        self,
        folium_map,
        report: ValidationReport,
        image_bounds: Optional[Tuple] = None,
    ):
        """
        FP (turuncu) ve FN (mavi) maskelerini Folium ImageOverlay olarak ekler.

        False Positive → turuncu : Bizim tespit ettiğimiz ama CEMS'in onaylamadığı
        False Negative → mavi    : CEMS'te var ama bizim kaçırdığımız
        """
        try:
            import folium
            import base64
            from PIL import Image
            import io

            if image_bounds is None:
                logger.warning("image_bounds yok — doğrulama overlay eklenemiyor")
                return

            lat_min, lon_min, lat_max, lon_max = image_bounds
            bounds = [[lat_min, lon_min], [lat_max, lon_max]]

            def _mask_to_rgba(mask: np.ndarray, rgb: Tuple) -> str:
                """Binary maske → RGBA PNG → base64."""
                h, w  = mask.shape
                rgba  = np.zeros((h, w, 4), dtype=np.uint8)
                where = mask > 0
                rgba[where, 0] = rgb[0]
                rgba[where, 1] = rgb[1]
                rgba[where, 2] = rgb[2]
                rgba[where, 3] = 160   # yarı saydam
                img = Image.fromarray(rgba, "RGBA")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return "data:image/png;base64," + \
                       base64.b64encode(buf.getvalue()).decode()

            if report.fp_mask is not None and report.fp_mask.any():
                folium.raster_layers.ImageOverlay(
                    image   = _mask_to_rgba(report.fp_mask, (247, 127, 0)),
                    bounds  = bounds,
                    name    = "🟠 False Positive (fazla tespit)",
                    opacity = 1.0,
                ).add_to(folium_map)

            if report.fn_mask is not None and report.fn_mask.any():
                folium.raster_layers.ImageOverlay(
                    image   = _mask_to_rgba(report.fn_mask, (0, 119, 182)),
                    bounds  = bounds,
                    name    = "🔵 False Negative (kaçırılan hasar)",
                    opacity = 1.0,
                ).add_to(folium_map)

            logger.debug("Doğrulama overlay'leri eklendi (FP/FN)")

        except Exception as exc:
            logger.error(f"Doğrulama overlay hatası: {exc}")

    # ── Mock / Fallback ─────────────────────────────────────

    @staticmethod
    def load_mock_features(
        mock_path: str = "data/mock_cems.geojson",
    ) -> List[CEMSFeature]:
        """
        Copernicus sunucusu erişilemez olduğunda yerel mock veriden yükler.

        Demo sırasında API yavaşlığı veya erişim izni sorunlarına karşı
        güvenlik ağı. data/mock_cems.geojson EMSR648 (2023 Türkiye depremi)
        temelli 15 gerçekçi bina/yol hasarı içerir.

        Args:
            mock_path: Mock GeoJSON dosyası yolu

        Returns:
            CEMSFeature listesi — gerçek WFS çıktısıyla aynı format
        """
        from pathlib import Path
        path = Path(mock_path)
        if not path.exists():
            logger.warning(f"Mock dosya bulunamadı: {mock_path}")
            return []
        try:
            with open(path, encoding="utf-8") as f:
                gj = json.load(f)
            svc = CEMSService()
            features = svc._parse_geojson(gj, layer_type="building_damage")
            logger.info(f"Mock CEMS verisi yüklendi: {len(features)} özellik — {mock_path}")
            return features
        except Exception as exc:
            logger.error(f"Mock yükleme hatası: {exc}")
            return []

    def fetch_wfs_features_safe(
        self,
        emsr_code:  str,
        layer_type: str = "building_damage",
        bbox:       Optional[Tuple[float, float, float, float]] = None,
        max_count:  int = 500,
        mock_path:  str = "data/mock_cems.geojson",
    ) -> Tuple[List["CEMSFeature"], bool]:
        """
        WFS çekme — başarısız olursa otomatik mock fallback.

        Demo sırasında sunucu sorunlarında patlamamak için
        bu metodu fetch_wfs_features() yerine kullanın.

        Returns:
            (features, is_live_data)
            is_live_data=True  → gerçek Copernicus verisi
            is_live_data=False → mock/demo verisi
        """
        try:
            features = self.fetch_wfs_features(
                emsr_code, layer_type, bbox, max_count
            )
            if features:
                return features, True
            # Sunucu boş döndürdü → mock'a düş
            logger.info("WFS boş yanıt → mock fallback")
        except Exception as exc:
            logger.warning(f"WFS başarısız ({exc}) → mock fallback")

        return self.load_mock_features(mock_path), False

    @staticmethod
    def generate_demo_features(
        center: Tuple[float, float],
        n: int = 20,
        radius_deg: float = 0.02,
    ) -> List[CEMSFeature]:
        """
        WFS erişilemez olduğunda demo özellikler üretir.
        Sunum sırasında gerçek CEMS verisi simüle eder.
        """
        rng      = np.random.default_rng(seed=42)
        features = []
        grades   = list(DAMAGE_GRADE_COLOR.keys())
        obj_types= ["building", "building", "building", "road"]

        for i in range(n):
            lat = center[0] + rng.uniform(-radius_deg, radius_deg)
            lon = center[1] + rng.uniform(-radius_deg, radius_deg)
            sz  = rng.uniform(0.0003, 0.001)
            grade    = grades[int(rng.integers(0, len(grades)))]
            obj_type = obj_types[int(rng.integers(0, len(obj_types)))]

            if obj_type == "road":
                geom = {
                    "type": "LineString",
                    "coordinates": [
                        [lon - sz, lat],
                        [lon + sz, lat],
                    ],
                }
                area = sz * 111320 * 6    # genişlik ~6m
            else:
                geom = {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon - sz, lat - sz],
                        [lon + sz, lat - sz],
                        [lon + sz, lat + sz],
                        [lon - sz, lat + sz],
                        [lon - sz, lat - sz],
                    ]],
                }
                area = (sz * 111320) ** 2

            features.append(CEMSFeature(
                feature_id   = f"DEMO_{i:04d}",
                geometry     = geom,
                damage_grade = grade,
                obj_type     = obj_type,
                area_m2      = float(area),
                centroid     = (lat, lon),
                properties   = {"source": "demo"},
            ))

        logger.info(f"Demo CEMS özellik üretildi: {len(features)} adet")
        return features

    # ── İç Metodlar ─────────────────────────────────────────

    def _parse_geojson(
        self, geojson: Dict, layer_type: str
    ) -> List[CEMSFeature]:
        """WFS GeoJSON yanıtını CEMSFeature listesine çevirir."""
        features = []
        # layer_type → default obj_type (properties'te yoksa kullanılır)
        obj_type_default = {
            "building_damage": "building",
            "road_damage":     "road",
            "flooded_area":    "area",
        }.get(layer_type, "area")

        for feat in geojson.get("features", []):
            try:
                props = feat.get("properties", {})
                geom  = feat.get("geometry", {})
                grade = (props.get("damage_grade") or
                         props.get("dmg_dscr")    or "Unknown")

                # obj_type: properties'teki değer varsa onu kullan (mock & gerçek veri uyumu)
                obj_type = (
                    props.get("obj_type") or          # mock_cems.geojson alanı
                    props.get("object_type") or        # CEMS resmi alan adı
                    props.get("obj_type_txt") or       # alternatif CEMS alanı
                    self._infer_obj_type(geom, obj_type_default)  # geometriye göre çıkarsama
                )

                # Centroid hesabı
                cen = self._geom_centroid(geom)
                if cen is None:
                    continue

                # Alan (m²)
                area = float(props.get("area_sqm") or
                             props.get("shape_area") or 0.0)

                features.append(CEMSFeature(
                    feature_id   = str(feat.get("id", f"F{len(features)}")),
                    geometry     = geom,
                    damage_grade = str(grade),
                    obj_type     = str(obj_type),
                    area_m2      = area,
                    centroid     = cen,
                    properties   = props,
                ))
            except Exception as exc:
                logger.debug(f"Feature parse hatası: {exc}")

        return features

    def _features_to_mask(
        self,
        features:      List[CEMSFeature],
        h: int, w: int,
        geo_transform,
        image_bounds:  Optional[Tuple],
    ) -> np.ndarray:
        """
        CEMSFeature listesini piksel maskesine dönüştürür.
        rasterio.features.rasterize kullanılır (CRS doğru korunur).
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        if not features:
            return mask

        try:
            import rasterio.features
            from shapely.geometry import shape as _shape

            if geo_transform is None and image_bounds is not None:
                from rasterio.transform import from_bounds
                lat_min, lon_min, lat_max, lon_max = image_bounds
                geo_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)

            if geo_transform is None:
                logger.warning("geo_transform yok — CEMS maskesi üretilemiyor")
                return mask

            shapes = []
            for feat in features:
                try:
                    geom = feat.geometry
                    if geom["type"] in ("Point",):
                        continue    # nokta → rasterize atlıyoruz
                    shp = _shape(geom)
                    if not shp.is_valid:
                        shp = shp.buffer(0)
                    shapes.append((geom, 1))
                except Exception:
                    continue

            if shapes:
                burned = rasterio.features.rasterize(
                    shapes,
                    out_shape=(h, w),
                    transform=geo_transform,
                    fill=0,
                    dtype=np.uint8,
                )
                mask = burned

        except ImportError:
            # rasterio yoksa → centroid piksel yaklaşımı
            logger.warning("rasterio yok — centroid piksel yaklaşımı kullanılıyor")
            if image_bounds:
                lat_min, lon_min, lat_max, lon_max = image_bounds
                for feat in features:
                    lat, lon = feat.centroid
                    px = int((lon - lon_min) / (lon_max - lon_min) * w)
                    py = int((lat_max - lat) / (lat_max - lat_min) * h)
                    if 0 <= px < w and 0 <= py < h:
                        # Yaklaşık alan büyüklüğünde daire
                        r_px = max(2, int((feat.area_m2 ** 0.5) / 2))
                        for dy in range(-r_px, r_px + 1):
                            for dx in range(-r_px, r_px + 1):
                                ny, nx_ = py + dy, px + dx
                                if 0 <= ny < h and 0 <= nx_ < w:
                                    mask[ny, nx_] = 1

        except Exception as exc:
            logger.error(f"CEMS maskesi üretim hatası: {exc}")

        return mask

    @staticmethod
    def _infer_obj_type(geom: Dict, default: str) -> str:
        """
        GeoJSON geometry tipine göre nesne türünü çıkarsıyor.
        LineString/MultiLineString → road, Polygon/MultiPolygon → building,
        Point → shelter. Bilinmiyorsa default döndür.
        """
        gtype = geom.get("type", "")
        if gtype in ("LineString", "MultiLineString"):
            return "road"
        if gtype in ("Polygon", "MultiPolygon"):
            return "building"
        if gtype == "Point":
            return "shelter"
        return default

    @staticmethod
    def _geom_centroid(geom: Dict) -> Optional[Tuple[float, float]]:
        """GeoJSON geometry'den centroid (lat, lon) döndürür."""
        try:
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "Point":
                return coords[1], coords[0]
            if gtype in ("LineString", "MultiPoint"):
                pts = coords
                return float(np.mean([p[1] for p in pts])), \
                       float(np.mean([p[0] for p in pts]))
            if gtype == "Polygon":
                ring = coords[0]
                return float(np.mean([p[1] for p in ring])), \
                       float(np.mean([p[0] for p in ring]))
            if gtype == "MultiPolygon":
                all_pts = [p for poly in coords for ring in poly for p in ring]
                return float(np.mean([p[1] for p in all_pts])), \
                       float(np.mean([p[0] for p in all_pts]))
        except Exception:
            pass
        return None

    @staticmethod
    def _static_layers(code: str, wms_url: str) -> List[CEMSLayer]:
        """Fallback: GetCapabilities başarısız olursa bilinen katman isimleri."""
        return [
            CEMSLayer(f"{code}_{lt}", label, wms_url, code)
            for lt, label in LAYER_TYPES.items()
        ]
