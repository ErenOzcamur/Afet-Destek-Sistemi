# ============================================================
# utils.py - GIS / Harita / Preprocessing / Loglama
# TUA Astro Hackathon DSS v2
# ============================================================

"""
Sorumluluklar:
  1. Loglama altyapısı (loguru)
  2. Görüntü I/O (GeoTIFF + standart formatlar)
  3. Preprocessing pipeline (normalize, denoise, CLAHE, align)
  4. GIS yardımcıları (piksel↔geo dönüşüm)

Tasarım ilkesi:
  - cv2 sadece bu modülde ve engine.py'de kullanılır.
  - main.py'den doğrudan cv2 import YAPILMAZ.
  - GeoTIFF meta (CRS, transform, bounds) her aşamada korunur.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config import app_config, cv_config

try:
    import rasterio
    from rasterio.transform import rowcol, xy
    _HAS_RASTERIO = True
except ImportError:
    rasterio = None
    rowcol = None
    xy = None
    _HAS_RASTERIO = False

# Preprocessing sabitleri
NLM_H = 10
NLM_H_COLOR = 10
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
ORB_MAX_FEATURES = 5000
MIN_MATCH_COUNT = 10
RANSAC_REPROJ_THRESHOLD = 5.0


# ────────────────────────────────────────────────────────────
# 1. LOGLAMA
# ────────────────────────────────────────────────────────────

def setup_logger(log_dir: Optional[str] = None) -> None:
    """Loguru tabanlı merkezi loglama başlatır."""
    log_dir = log_dir or app_config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()

    logger.add(
        sys.stderr, level=app_config.log_level, colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
            "<cyan>{module}:{function}:{line}</cyan> - <level>{message}</level>"
        ),
    )
    logger.add(
        os.path.join(log_dir, "dss_{time:YYYY-MM-DD}.log"),
        level=app_config.log_level,
        rotation=app_config.log_rotation,
        retention="7 days",
        encoding="utf-8",
    )
    logger.info("Logger başlatıldı.")


# ────────────────────────────────────────────────────────────
# 2. GÖRÜNTÜ I/O
# ────────────────────────────────────────────────────────────

class GeoImageReader:
    """GeoTIFF ve standart görüntü okuyucu."""

    @staticmethod
    def read(filepath) -> Dict:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

        ext = filepath.suffix.lower()
        if ext in (".tif", ".tiff"):
            return GeoImageReader._read_geotiff(filepath)
        return GeoImageReader._read_standard(filepath)

    @staticmethod
    def _read_geotiff(filepath: Path) -> Dict:
        if not _HAS_RASTERIO:
            logger.warning("rasterio yok, standart okuma ile devam.")
            return GeoImageReader._read_standard(filepath)

        try:
            with rasterio.open(filepath) as src:
                if src.count >= 3:
                    image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                else:
                    image = src.read(1)
                meta = {
                    "crs": str(src.crs) if src.crs else None,
                    "transform": src.transform,
                    "bounds": src.bounds,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": str(src.dtypes[0]),
                }
        except rasterio.errors.RasterioIOError as e:
            logger.error(f"GeoTIFF okunamadı: {e}")
            raise
        logger.info(f"GeoTIFF yüklendi: {filepath.name} | CRS={meta['crs']}")
        return {"image": image, "meta": meta}

    @staticmethod
    def _read_standard(filepath: Path) -> Dict:
        image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Görüntü okunamadı: {filepath}")
        if len(image.shape) == 3 and image.shape[2] >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        meta = {"width": image.shape[1], "height": image.shape[0]}
        logger.info(f"Görüntü yüklendi: {filepath.name} | {meta['width']}x{meta['height']}")
        return {"image": image, "meta": meta}


# ────────────────────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Adımlar: dtype normalize → resize → denoise → CLAHE
    Georeferencing meta DEĞİŞMEZ, sadece piksel değerleri işlenir.
    """

    def __init__(self, config=None):
        self.cfg = config or cv_config

    def run(self, image: np.ndarray, denoise=True, equalize=True) -> np.ndarray:
        logger.info(f"Pipeline başlıyor | shape={image.shape} dtype={image.dtype}")
        out = self._normalize_dtype(image)
        out = self._resize(out)
        if denoise:
            out = self._denoise(out)
        if equalize:
            out = self._equalize(out)
        logger.info(f"Pipeline bitti | shape={out.shape}")
        return out

    def _normalize_dtype(self, img):
        if img.dtype == np.uint8:
            return img
        if img.dtype in (np.uint16, np.int16):
            mn, mx = img.min(), img.max()
            if mx - mn == 0:
                return np.zeros_like(img, dtype=np.uint8)
            return ((img.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
        if np.issubdtype(img.dtype, np.floating):
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img.astype(np.uint8)

    def _resize(self, img, target=None):
        h, w = img.shape[:2]
        mx = self.cfg.max_processing_dim
        if target:
            return cv2.resize(img, target, interpolation=cv2.INTER_AREA)
        if max(h, w) <= mx:
            return img
        s = mx / max(h, w)
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    def _denoise(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            try:
                return cv2.fastNlMeansDenoisingColored(
                    img, None, NLM_H, NLM_H_COLOR,
                    NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW,
                )
            except cv2.error as e:
                logger.warning(f"NlMeans denoise başarısız, GaussianBlur fallback: {e}")
        return cv2.GaussianBlur(img, self.cfg.blur_kernel_size, 0)

    def _equalize(self, img):
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        if len(img.shape) == 3:
            l_channel, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
            lab = cv2.merge([clahe.apply(l_channel), a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return clahe.apply(img)

    @staticmethod
    def align_images(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """ORB + Homography ile görüntü hizalama."""
        ref_g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) if len(ref.shape) == 3 else ref
        tgt_g = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY) if len(tgt.shape) == 3 else tgt

        orb = cv2.ORB_create(ORB_MAX_FEATURES)
        kp1, d1 = orb.detectAndCompute(ref_g, None)
        kp2, d2 = orb.detectAndCompute(tgt_g, None)
        if d1 is None or d2 is None:
            logger.warning("Feature bulunamadı, hizalama atlanıyor.")
            return tgt

        matches = sorted(
            cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
            key=lambda m: m.distance,
        )
        if len(matches) < MIN_MATCH_COUNT:
            logger.warning(f"Yetersiz eşleşme ({len(matches)} < {MIN_MATCH_COUNT}), atlanıyor.")
            return tgt

        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst, src, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if M is None:
            logger.warning("Homography bulunamadı, hizalama atlanıyor.")
            return tgt
        return cv2.warpPerspective(tgt, M, (ref.shape[1], ref.shape[0]))


# ────────────────────────────────────────────────────────────
# 4. GIS YARDIMCILARI
# ────────────────────────────────────────────────────────────

def pixel_to_geo(px_x: float, px_y: float, transform) -> Optional[Tuple[float, float]]:
    """Piksel koordinatını (lat, lon)'a dönüştürür."""
    if transform is None or not _HAS_RASTERIO:
        if not _HAS_RASTERIO:
            logger.warning("pixel_to_geo başarısız: rasterio kurulu değil.")
        return None
    try:
        geo_x, geo_y = xy(transform, int(px_y), int(px_x))
        return (geo_y, geo_x)  # (lat, lon)
    except Exception as e:
        logger.warning(f"pixel_to_geo başarısız: {e}")
        return None


def geo_to_pixel(lat: float, lon: float, transform) -> Optional[Tuple[int, int]]:
    """Geo koordinatı piksel konumuna dönüştürür."""
    if transform is None or not _HAS_RASTERIO:
        if not _HAS_RASTERIO:
            logger.warning("geo_to_pixel başarısız: rasterio kurulu değil.")
        return None
    try:
        row, col = rowcol(transform, lon, lat)
        return (col, row)
    except Exception as e:
        logger.warning(f"geo_to_pixel başarısız: {e}")
        return None
