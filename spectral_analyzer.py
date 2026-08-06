# ============================================================
# spectral_analyzer.py - Zaman Serisi & Spektral Analiz
# TUA Astro Hackathon DSS — Planet Multi-Band + Change Detection
# ============================================================

"""
İki teknik:

1. Zaman Serisi / Change Detection:
   Bir yolun dokusu (piksel renk değeri) günden güne değişiyorsa
   kapanma erken tespit edilir. Planet'in günlük revisit kapasitesi
   bu analizi mümkün kılar.

   Algoritma:
     - n görüntü dizisi → her pikselde z-score hesabı
     - |z| > threshold → anomali (yol kapanmaya başlıyor)
     - CUSUM (Cumulative Sum) → değişim başlangıç zamanını saptar

2. Spektral Analiz (NDVI, NDWI, NDBI):
   Planet 4-bant: Blue(B1), Green(B2), Red(B3), NIR(B4)

   NDVI = (NIR - Red) / (NIR + Red)   → bitki örtüsü kapanması
   NDWI = (Green - NIR) / (Green + NIR) → su baskını tespiti
   NDBI = (SWIR - NIR) / (SWIR + NIR) → yapılı alan değişimi

   Çıplak gözle seçilemeyen kapalılık durumları bu bantlarla
   ortaya çıkar (örn: yolu kaplayan yeşillik, sığ su birikintisi).

   Not: Planet PlanetScope 4-bant sağlar (Blue/Green/Red/NIR).
        SWIR sadece Planet SuperDove (8-bant) veya Sentinel-2'de.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ── Sabitler ────────────────────────────────────────────────

# Planet bant sırası (1-indexed rasterio → 0-indexed numpy)
BAND_BLUE  = 0   # B1
BAND_GREEN = 1   # B2
BAND_RED   = 2   # B3
BAND_NIR   = 3   # B4 (PlanetScope 4-bant)

# Spektral indeks eşikleri
NDVI_VEGETATION_THR = 0.3    # > 0.3 → bitki örtüsü
NDWI_WATER_THR      = 0.1    # > 0.1 → su birikintisi
NDBI_BUILT_THR      = 0.0    # > 0.0 → yapılı alan baskın

# Z-score eşiği (change detection)
ZSCORE_ANOMALY_THR  = 2.5

# Normalizasyon sabitleri
UINT16_REFLECTANCE_SCALE = 10000.0
UINT8_SCALE              = 255.0

# Sayısal koruma sabitleri
EPSILON           = 1e-6
MIN_BASELINE_STD  = 1.0

# Zaman serisi sabitleri
BASELINE_RATIO    = 0.6
MAX_CHANGE_EVENTS = 50
SEVERITY_HIGH_Z   = 4.0
SEVERITY_MEDIUM_Z = 3.0


# ── Dataclass'lar ────────────────────────────────────────────

@dataclass
class SpectralResult:
    """Tek görüntü spektral analiz sonucu."""
    ndvi:           np.ndarray     # (H, W) float — bitki örtüsü
    ndwi:           np.ndarray     # (H, W) float — su
    ndbi:           np.ndarray     # (H, W) float — yapılı alan
    vegetation_mask: np.ndarray   # (H, W) bool
    water_mask:     np.ndarray    # (H, W) bool
    built_mask:     np.ndarray    # (H, W) bool
    stats:          Dict          = field(default_factory=dict)

    def road_obstruction_mask(self) -> np.ndarray:
        """
        Yol kapanma maskesi: bitki VEYA su kaplayan pikseller.
        Bu pikseller üzerindeki yol segmentleri → RESTRICTED/CLOSED.
        """
        return (self.vegetation_mask | self.water_mask).astype(np.uint8) * 255


@dataclass
class ChangeEvent:
    """Tek bir zaman serisi anomali olayı."""
    frame_index:    int            # Zaman serisindeki görüntü sırası
    pixel_y:        int
    pixel_x:        int
    z_score:        float
    cusum_value:    float
    change_type:    str            # "increase" | "decrease"
    severity:       str            # "low" | "medium" | "high"


@dataclass
class TimeSeriesResult:
    """Zaman serisi analiz sonucu."""
    n_frames:           int
    anomaly_mask:       np.ndarray   # (H, W) — anomali piksel haritası
    change_events:      List[ChangeEvent]
    first_change_frame: int          # CUSUM ile saptanan değişim başlangıcı
    change_score:       float        # 0–1, toplam değişim şiddeti
    stats:              Dict         = field(default_factory=dict)


# ── Spektral Analiz ──────────────────────────────────────────

class SpectralAnalyzer:
    """
    Planet 4-bant görüntüden spektral indeks hesabı.

    0.5m/px avantajı:
      - Yolu kaplayan 2m² su birikintisi = 8 piksel → tespit edilebilir
      - 2.5m/px'de aynı birikim 0.3 piksel → alt-piksel gürültüsüne gömülür
    """

    def __init__(self, image_4band: np.ndarray):
        """
        Args:
            image_4band: (H, W, 4) veya (4, H, W) float32/uint16 array
                        Bant sırası: Blue, Green, Red, NIR
        """
        if image_4band.ndim != 3:
            raise ValueError("3 boyutlu (H, W, C) veya (C, H, W) array bekleniyor")

        if image_4band.shape[0] == 4:
            # (4, H, W) → (H, W, 4)
            image_4band = np.moveaxis(image_4band, 0, -1)

        if image_4band.shape[2] < 3:
            raise ValueError("En az 3 bant gereklidir (Blue, Green, Red)")

        # uint16 → float [0, 1] normalize
        if image_4band.dtype == np.uint16:
            self._img = image_4band.astype(np.float32) / UINT16_REFLECTANCE_SCALE
        elif image_4band.dtype == np.uint8:
            self._img = image_4band.astype(np.float32) / UINT8_SCALE
        else:
            self._img = image_4band.astype(np.float32)

        self._h, self._w = self._img.shape[:2]
        self._has_nir    = self._img.shape[2] >= 4
        logger.info(
            f"SpectralAnalyzer: {self._h}×{self._w}px, "
            f"{'4' if self._has_nir else '3'}-bant"
        )

    def analyze(self) -> SpectralResult:
        """Tüm spektral indeksleri hesaplar."""
        b = self._img[:, :, BAND_BLUE]
        g = self._img[:, :, BAND_GREEN]
        r = self._img[:, :, BAND_RED]
        nir = self._img[:, :, BAND_NIR] if self._has_nir else r * 0.8

        ndvi = self._safe_index(nir, r)   # (NIR-R)/(NIR+R)
        ndwi = self._safe_index(g, nir)   # (G-NIR)/(G+NIR)
        ndbi = self._safe_index(r, nir)   # (R-NIR)/(R+NIR) — yaklaşık NDBI

        veg_mask   = ndvi > NDVI_VEGETATION_THR
        water_mask = ndwi > NDWI_WATER_THR
        built_mask = ndbi > NDBI_BUILT_THR

        stats = {
            "ndvi_mean":   float(np.nanmean(ndvi)),
            "ndvi_max":    float(np.nanmax(ndvi)),
            "ndwi_mean":   float(np.nanmean(ndwi)),
            "water_pct":   float(water_mask.mean() * 100),
            "veg_pct":     float(veg_mask.mean() * 100),
            "built_pct":   float(built_mask.mean() * 100),
        }
        logger.info(
            f"Spektral analiz: NDVI_mean={stats['ndvi_mean']:.3f}, "
            f"su={stats['water_pct']:.1f}%, bitki={stats['veg_pct']:.1f}%"
        )

        return SpectralResult(
            ndvi=ndvi, ndwi=ndwi, ndbi=ndbi,
            vegetation_mask=veg_mask,
            water_mask=water_mask,
            built_mask=built_mask,
            stats=stats,
        )

    def ndvi_colormap(self, ndvi: np.ndarray) -> np.ndarray:
        """
        NDVI değerlerini RGB renk haritasına dönüştürür.
        -1 → kırmızı (su/yapı), 0 → sarı (çıplak toprak), +1 → koyu yeşil (bitki)
        """
        return self._apply_colormap(ndvi, "COLORMAP_RdYlGn", fallback_channel=1)

    def ndwi_colormap(self, ndwi: np.ndarray) -> np.ndarray:
        """NDWI → mavi tonlama (su bölgeleri mavi)."""
        return self._apply_colormap(ndwi, "COLORMAP_OCEAN", fallback_channel=2)

    @staticmethod
    def _apply_colormap(
        index_arr: np.ndarray,
        cv2_colormap_name: str,
        fallback_channel: int,
    ) -> np.ndarray:
        """[-1, 1] indeks → RGB renk haritası; cv2 yoksa basit tek kanal fallback."""
        normalized = np.clip((index_arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
        try:
            import cv2
            colormap = cv2.applyColorMap(normalized, getattr(cv2, cv2_colormap_name))
            return cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        except ImportError:
            logger.warning("cv2 bulunamadı, basit renk haritası fallback'i kullanılıyor")
            rgb = np.zeros((*index_arr.shape, 3), dtype=np.uint8)
            rgb[:, :, fallback_channel] = np.clip(
                (index_arr * 127 + 127), 0, 255
            ).astype(np.uint8)
            return rgb

    @staticmethod
    def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(A - B) / (A + B) — sıfıra bölme koruması."""
        denom = a + b
        out = np.zeros_like(a, dtype=np.float32)
        np.divide(a - b, denom, out=out, where=np.abs(denom) > EPSILON)
        return out


# ── Zaman Serisi Analizi ─────────────────────────────────────

class TimeSeriesAnalyzer:
    """
    n görüntü dizisinde piksel bazlı değişim tespiti.

    Planet günlük revisit → belirli bir yol üzerindeki piksellerin
    gri değeri (veya NDVI'sı) izlenerek kapanma başlangıcı saptanır.

    Algoritma:
      1. Her piksel için baseline mean ± std hesapla
      2. Yeni görüntüde z-score = (yeni - mean) / std
      3. |z| > 2.5 → anomali
      4. CUSUM: cumulative sum of standardized residuals
         → değişimin tam başladığı frame'i saptar

    0.5m/px avantajı:
      Moloz dökülmesi 0.5m² → 2 piksel anomali.
      2.5m/px'de aynı olay 0.08 piksel → z-score eşiğine ulaşmaz.
    """

    def __init__(self, images: List[np.ndarray]):
        """
        Args:
            images: Kronolojik sıralı görüntü listesi.
                   Her biri (H, W) gri veya (H, W, C) renkli.
                   Tüm görüntüler aynı boyutta olmalı.
        """
        if len(images) < 2:
            raise ValueError("Zaman serisi analizi için en az 2 görüntü gereklidir.")

        for i, img in enumerate(images):
            if img.ndim not in (2, 3):
                raise ValueError(f"Görüntü {i}: 2B veya 3B array bekleniyor")

        # Gri tonlamaya dönüştür
        grays = []
        for img in images:
            if img.ndim == 3:
                g = img.mean(axis=2).astype(np.float32)
            else:
                g = img.astype(np.float32)
            grays.append(g)

        # Boyut standardizasyonu
        h0, w0 = grays[0].shape
        self._stack = np.stack([
            self._resize_if_needed(g, h0, w0) for g in grays
        ], axis=0)   # (T, H, W)

        self._T, self._H, self._W = self._stack.shape
        logger.info(
            f"TimeSeriesAnalyzer: {self._T} frame, "
            f"{self._H}×{self._W}px"
        )

    def analyze(
        self,
        baseline_frames:   Optional[int] = None,
        zscore_threshold:  float = ZSCORE_ANOMALY_THR,
        cusum_k:           float = 0.5,
        cusum_h:           float = 5.0,
    ) -> TimeSeriesResult:
        """
        Tam zaman serisi analizi çalıştırır.

        Args:
            baseline_frames : Baseline için ilk N frame (None → ilk %60)
            zscore_threshold: Anomali z-score eşiği
            cusum_k         : CUSUM slack parametresi
            cusum_h         : CUSUM karar eşiği

        Returns:
            TimeSeriesResult
        """
        if baseline_frames is not None and baseline_frames <= 0:
            raise ValueError("baseline_frames pozitif bir tamsayı olmalıdır")

        n_base = (
            baseline_frames if baseline_frames is not None
            else max(1, int(self._T * BASELINE_RATIO))
        )
        n_base = min(n_base, self._T - 1)

        baseline  = self._stack[:n_base]
        bl_mean   = baseline.mean(axis=0).astype(np.float32)    # (H, W)
        bl_std    = np.maximum(
            baseline.std(axis=0).astype(np.float32), MIN_BASELINE_STD
        )

        # Z-score haritası — frame-frame artımlı hesap (bellek dostu)
        anomaly_z = np.zeros((self._H, self._W), dtype=np.float32)
        for t in range(n_base, self._T):
            z_frame = (self._stack[t] - bl_mean) / bl_std
            anomaly_z = np.maximum(anomaly_z, np.abs(z_frame))
        anomaly_mask = (anomaly_z > zscore_threshold).astype(np.uint8) * 255

        # CUSUM — değişim başlangıcı
        first_change, cusum_score = self._cusum(
            bl_mean, bl_std, n_base, cusum_k, cusum_h
        )

        # Anomali olay listesi
        events = self._extract_events(bl_mean, bl_std, n_base, zscore_threshold)

        change_score = float(np.mean(anomaly_mask > 0))

        stats = {
            "n_frames":           self._T,
            "baseline_frames":    n_base,
            "anomaly_pixel_pct":  round(change_score * 100, 2),
            "max_zscore":         float(anomaly_z.max()),
            "mean_zscore":        float(anomaly_z.mean()),
            "first_change_frame": first_change,
        }
        logger.info(
            f"TS analizi: anomali={stats['anomaly_pixel_pct']:.1f}%, "
            f"max_z={stats['max_zscore']:.2f}, "
            f"değişim frame={first_change}"
        )

        return TimeSeriesResult(
            n_frames           = self._T,
            anomaly_mask       = anomaly_mask,
            change_events      = events,
            first_change_frame = first_change,
            change_score       = change_score,
            stats              = stats,
        )

    def pixel_timeseries(
        self, y: int, x: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Belirli bir piksel için zaman serisi döndürür.

        Returns:
            (values, z_scores) her biri (T,) şeklinde
        """
        vals    = self._stack[:, y, x]
        baseline= vals[:max(1, len(vals) // 2)]
        mean    = baseline.mean()
        std     = max(MIN_BASELINE_STD, baseline.std())
        zscores = (vals - mean) / std
        return vals, zscores

    # ── İç Metodlar ─────────────────────────────────────────

    def _cusum(
        self,
        bl_mean: np.ndarray,
        bl_std:  np.ndarray,
        n_base:  int,
        k: float, h: float,
    ) -> Tuple[int, float]:
        """
        CUSUM (Page, 1954) ile değişim başlangıç frame tespiti.
        Her frame için ortalama CUSUM değeri hesaplanır.
        Eşiği aşan ilk frame = değişim başlangıcı.
        """
        cusum_pos = np.zeros((self._H, self._W), dtype=np.float32)
        cusum_neg = np.zeros((self._H, self._W), dtype=np.float32)
        max_cusum = 0.0
        first_change = self._T - 1

        for t in range(n_base, self._T):
            z = (self._stack[t] - bl_mean) / bl_std
            cusum_pos = np.maximum(0, cusum_pos + z - k)
            cusum_neg = np.minimum(0, cusum_neg + z + k)
            mean_cs   = float(np.mean(np.maximum(cusum_pos, -cusum_neg)))

            if mean_cs > h and mean_cs > max_cusum:
                max_cusum    = mean_cs
                first_change = t

        return first_change, max_cusum

    def _extract_events(
        self,
        bl_mean:   np.ndarray,
        bl_std:    np.ndarray,
        n_base:    int,
        threshold: float,
        max_events: int = MAX_CHANGE_EVENTS,
    ) -> List[ChangeEvent]:
        """
        Eşiği aşan ilk pikselleri (satır sırasıyla) olay listesine dönüştürür,
        ardından |z|'ye göre büyükten küçüğe sıralar.
        """
        events = []
        for t in range(n_base, self._T):
            z_frame = (self._stack[t] - bl_mean) / bl_std
            ys, xs  = np.where(np.abs(z_frame) > threshold)
            for y, x in zip(ys[:max_events], xs[:max_events]):
                z_val = float(z_frame[y, x])
                events.append(ChangeEvent(
                    frame_index = t,
                    pixel_y     = int(y),
                    pixel_x     = int(x),
                    z_score     = z_val,
                    cusum_value = 0.0,
                    change_type = "increase" if z_val > 0 else "decrease",
                    severity    = ("high"   if abs(z_val) > SEVERITY_HIGH_Z else
                                   "medium" if abs(z_val) > SEVERITY_MEDIUM_Z
                                   else "low"),
                ))
            if len(events) >= max_events:
                break

        return sorted(events, key=lambda e: abs(e.z_score), reverse=True)[:max_events]

    @staticmethod
    def _resize_if_needed(
        img: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Görüntü boyutunu hedef boyuta getirir."""
        if img.shape == (h, w):
            return img

        logger.warning(f"Frame {img.shape} -> ({h},{w}) yeniden boyutlandırıldı")
        try:
            from PIL import Image as _PIL
            clipped = np.clip(img, 0, 255).astype(np.uint8)
            resized = _PIL.fromarray(clipped).resize((w, h))
            return np.array(resized).astype(np.float32)
        except ImportError:
            logger.warning("PIL bulunamadı, kırp/doldur fallback'i kullanılıyor")
            pad_h = max(0, h - img.shape[0])
            pad_w = max(0, w - img.shape[1])
            if pad_h == 0 and pad_w == 0:
                return img[:h, :w]
            padded = np.pad(img, ((0, pad_h), (0, pad_w)))
            return padded[:h, :w]


# ── Yardımcı: RGB'den Sahte 4-Bant ──────────────────────────

def rgb_to_pseudo_4band(rgb: np.ndarray) -> np.ndarray:
    """
    3-bantlı RGB görüntüyü sahte 4-bantlı (Blue/Green/Red/NIR) diziye çevirir.

    Planet verisi yoksa demo modunda çalışmak için kullanılır.
    NIR tahmini: kırmızı bandın yansıması + yeşilin %20'si (yaklaşık).
    Not: Gerçek NIR değil, yalnızca spektral analiz yapısını göstermek için.
    """
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # Sahte NIR: bitki alanlarında yeşil yansıması daha yüksek olduğundan
    # NIR ≈ Green * 1.3 + Red * 0.2 (kaba yaklaşım)
    nir = np.clip(g * 1.3 + r * 0.2, 0, 255).astype(np.float32)

    return np.stack([b, g, r, nir], axis=-1)
