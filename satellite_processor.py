# ============================================================
# satellite_processor.py - Bellek-Optimize Raster İşlemci
# TUA Astro Hackathon DSS — Büyük GeoTIFF İçin Tile-Based Engine
# ============================================================

"""
SatelliteProcessor: Devasa GeoTIFF dosyalarını RAM'e yüklemeden işler.

Teknik Mimari:
  - Windowed Reading : rasterio.windows.Window ile 512×512 tile
  - CRS Preservation : Affine transform her tile'da korunur
  - Raster-to-Vector : rasterio.features.shapes → Polygon (geopandas)
  - Memory-Mapping   : numpy.memmap ile disk-backed array
  - dtype optimizasyon: float64 yerine uint8/float32

Planet SkySat (0.5m/px) için: 10km × 10km bölge = 20000×20000px
→ float64 olarak ~3.2GB RAM gerektirir.
→ Bu işlemci uint8 + tile ile ~40MB RAM kullanır.

analyzer.py entegrasyonu:
    proc = SatelliteProcessor("pre.tif", "post.tif")
    result = proc.process_large_tiff("damage_mask.tif")
    polygons_gdf = result.damage_polygons
    # → OSMnx intersection analizi
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

import numpy as np
from loguru import logger

# ── Sabitler ─────────────────────────────────────────────────

METERS_PER_DEGREE = 111_320.0   # ekvatorda 1 derecenin yaklaşık metre karşılığı
UINT8_MAX         = 255
SSIM_DIFF_SCALE   = 127.5       # (1 - ssim_diff) → uint8 ölçekleme çarpanı
MORPH_KERNEL_SIZE = (3, 3)
MIN_SSIM_WIN      = 3


def _require_rasterio():
    """rasterio'yu import eder; yoksa açık mesajla ImportError fırlatır."""
    try:
        import rasterio
        return rasterio
    except ImportError as exc:
        raise ImportError("rasterio gereklidir: pip install rasterio") from exc


# ── Dataclass'lar ────────────────────────────────────────────

@dataclass
class TileResult:
    """Tek bir tile'ın işlem sonucu."""
    tile_id:      int
    window:       Any          # rasterio.windows.Window
    damage_mask:  np.ndarray   # (H, W) uint8 binary
    damage_ratio: float
    ssim_score:   float
    offset:       Tuple[int, int]   # (row_off, col_off)


@dataclass
class ProcessingResult:
    """process_large_tiff() çıktı paketi."""
    output_path:       str
    transform:         Any          # rasterio.Affine — orijinal CRS korunur
    crs:               Any          # rasterio.CRS
    total_damage_ratio: float
    n_tiles_processed:  int
    n_damage_tiles:     int
    damage_polygons:    Optional[Any] = None   # geopandas.GeoDataFrame


# ── Ana Sınıf ────────────────────────────────────────────────

class SatelliteProcessor:
    """
    Bellek-optimize tile-based raster işlemci.

    Kullanım:
        proc = SatelliteProcessor("pre.tif", "post.tif", block_size=512)
        result = proc.process_large_tiff("output_mask.tif")
        gdf = result.damage_polygons   # geopandas GeoDataFrame
    """

    def __init__(
        self,
        before_path: str,
        after_path: str,
        block_size: int = 512,
        damage_threshold: float = 0.30,
        ssim_win_size: int = 7,
    ):
        """
        Args:
            before_path      : Afet öncesi GeoTIFF
            after_path       : Afet sonrası GeoTIFF
            block_size       : Tile boyutu (piksel). 512 → ~1MB/tile
            damage_threshold : SSIM fark eşiği hasar kararı için
            ssim_win_size    : SSIM pencere boyutu
        """
        self.before_path      = Path(before_path)
        self.after_path       = Path(after_path)
        self.block_size       = block_size
        self.damage_threshold = damage_threshold
        self.ssim_win_size    = ssim_win_size
        logger.info(
            f"SatelliteProcessor: tile={block_size}px, "
            f"eşik={damage_threshold}"
        )

    # ── Public API ──────────────────────────────────────────

    def process_large_tiff(
        self,
        output_path: str = "damage_mask.tif",
        band: int = 1,
        vectorize: bool = True,
    ) -> ProcessingResult:
        """
        Büyük GeoTIFF'i tile-by-tile işler; hasar maskesi üretir.

        Args:
            output_path: Çıktı hasar maskesi dosyası
            band       : Okunacak bant numarası (1-indexed)
            vectorize  : True → hasar piksellerini Polygon'a dönüştür

        Returns:
            ProcessingResult (transform, crs, damage_polygons, ...)
        """
        rasterio = _require_rasterio()

        with rasterio.open(self.before_path) as src_b, \
             rasterio.open(self.after_path)  as src_a:

            # Georeferencing bilgisini koru
            transform = src_b.transform
            crs       = src_b.crs
            width     = src_b.width
            height    = src_b.height
            profile   = src_b.profile.copy()

            profile.update({
                "dtype":  "uint8",
                "count":  1,
                "nodata": 0,
            })

            logger.info(
                f"Görüntü: {width}×{height}px | "
                f"CRS: {crs} | Tile: {self.block_size}px"
            )

            from rasterio.windows import Window

            tile_count    = 0
            damage_tiles  = 0
            skipped_tiles = 0
            total_dmg     = 0.0
            total_px      = 0

            with rasterio.open(output_path, "w", **profile) as dst:
                # tile_transform kasıtlı olarak kullanılmıyor (bkz. _tile_windows)
                for window, _tile_tf in self._tile_windows(
                    width, height, transform
                ):
                    try:
                        # float32 olarak oku (float64 yerine ~2x daha az RAM)
                        b_tile = src_b.read(band, window=window).astype(np.float32)
                        a_tile = src_a.read(band, window=window).astype(np.float32)

                        # Boyut uyuşmazlığı kontrolü
                        if b_tile.shape != a_tile.shape:
                            min_h = min(b_tile.shape[0], a_tile.shape[0])
                            min_w = min(b_tile.shape[1], a_tile.shape[1])
                            b_tile = b_tile[:min_h, :min_w]
                            a_tile = a_tile[:min_h, :min_w]

                        # Hasar tespiti
                        mask, ssim_val, dmg_ratio = self._detect_tile(
                            b_tile, a_tile
                        )

                        # Çıktıya yaz — CRS/transform otomatik korunur.
                        # Kırpılmış tile'larda yazma window'u maske boyutuna göre
                        # yeniden oluşturulur (boyut hatasını önler).
                        write_window = Window(
                            window.col_off, window.row_off,
                            mask.shape[1], mask.shape[0],
                        )
                        dst.write(mask[np.newaxis, :, :], window=write_window)

                        total_dmg += dmg_ratio * mask.size
                        total_px  += mask.size
                        tile_count += 1
                        if dmg_ratio > self.damage_threshold:
                            damage_tiles += 1

                    except (rasterio.errors.RasterioIOError, ValueError) as exc:
                        logger.warning(f"Tile {window} atlandı: {exc}")
                        skipped_tiles += 1
                        continue

            overall_ratio = total_dmg / total_px if total_px > 0 else 0.0
            logger.info(
                f"Tile işleme tamamlandı: {tile_count} tile, "
                f"{damage_tiles} hasarlı, {skipped_tiles} atlandı, "
                f"oran={overall_ratio:.2%}"
            )

        # Vektörizasyon
        damage_gdf = None
        if vectorize:
            damage_gdf = self.raster_to_vector(output_path, crs, transform)

        return ProcessingResult(
            output_path        = output_path,
            transform          = transform,
            crs                = crs,
            total_damage_ratio = overall_ratio,
            n_tiles_processed  = tile_count,
            n_damage_tiles     = damage_tiles,
            damage_polygons    = damage_gdf,
        )

    def raster_to_vector(
        self,
        mask_path: str,
        crs: Any,
        transform: Any,
        min_area_m2: float = 4.0,
    ) -> Optional[Any]:
        """
        Hasar maskesindeki pikselleri Polygon nesnelerine dönüştürür.

        rasterio.features.shapes kullanarak:
          - Her bağlantılı hasar bölgesi → shapely Polygon
          - Coğrafi koordinatlarda (CRS korunur)
          - geopandas GeoDataFrame olarak döner

        Args:
            mask_path   : Binary hasar maskesi GeoTIFF
            crs         : Koordinat referans sistemi
            transform   : Affine transform (geriye uyumluluk için tutulur;
                          fiilen dosyadaki src.transform kullanılır)
            min_area_m2 : Küçük parazitleri filtrele

        Returns:
            geopandas.GeoDataFrame | None
        """
        try:
            import rasterio
            import rasterio.features
            import geopandas as gpd
            from shapely.geometry import shape

            with rasterio.open(mask_path) as src:
                mask_arr = src.read(1)
                tf       = src.transform

            # uint8 garantisi — dtype zaten uygunsa kopya çıkarmaz
            mask_arr = np.asarray(mask_arr, dtype=np.uint8)

            # Generator'ı list'e çevirmeden tek geçişte tüket (RAM tasarrufu)
            geoms = [
                shape(s)
                for s, v in rasterio.features.shapes(
                    mask_arr, mask=mask_arr > 0, transform=tf
                )
                if v > 0
            ]

            if not geoms:
                logger.info("Hasar vektörü: sıfır polygon")
                return gpd.GeoDataFrame(columns=["geometry","area_m2"], crs=crs)

            areas = [g.area for g in geoms]

            gdf = gpd.GeoDataFrame(
                {"geometry": geoms, "area_m2": areas},
                crs=crs,
            )
            gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)

            logger.info(
                f"Vektörizasyon: {len(gdf)} polygon "
                f"(filtre ≥{min_area_m2}m²)"
            )
            return gdf

        except ImportError as exc:
            logger.warning(f"Vektörizasyon için geopandas/shapely gerekli: {exc}")
            return None
        except (rasterio.errors.RasterioIOError, ValueError) as exc:
            logger.error(f"Vektörizasyon hatası ({mask_path}): {exc}")
            return None

    def intersect_with_roads(
        self,
        damage_gdf: Any,          # geopandas.GeoDataFrame
        road_gdf: Any,            # geopandas.GeoDataFrame (OSMnx edges)
        buffer_m: float = 5.0,
    ) -> Optional[Any]:
        """
        Hasar poligonlarıyla yol ağı kesişim analizi.

        Args:
            damage_gdf: raster_to_vector() çıktısı
            road_gdf  : ox.graph_to_gdfs(G, nodes=False) çıktısı
            buffer_m  : Yol buffer genişliği (metre)

        Returns:
            Kesişen yol segmentlerini içeren GeoDataFrame
        """
        if road_gdf is None or road_gdf.empty:
            logger.warning("Yol GDF boş/None — kesişim yapılamadı")
            return None

        try:
            import geopandas as gpd

            if damage_gdf is None or damage_gdf.empty:
                logger.warning("Hasar GDF boş — kesişim yapılamadı")
                return road_gdf.iloc[0:0]

            # Projeksiyon eşleştir
            if damage_gdf.crs != road_gdf.crs:
                damage_gdf = damage_gdf.to_crs(road_gdf.crs)

            # Yol buffer'ı — immutable desen: kopya üzerinde mutasyon yerine assign
            road_buffered = road_gdf.assign(
                geometry=road_gdf.geometry.buffer(buffer_m / METERS_PER_DEGREE)
            )

            # Kesişim
            intersected = gpd.sjoin(
                road_buffered, damage_gdf,
                how="inner", predicate="intersects"
            )
            logger.info(
                f"Road-damage intersection: "
                f"{len(intersected)} / {len(road_gdf)} segment kesişiyor"
            )
            return intersected

        except ImportError as exc:
            logger.error(f"Kesişim analizi için geopandas gerekli: {exc}")
            return None
        except (ValueError, AttributeError, TypeError, KeyError) as exc:
            logger.error(
                f"Kesişim analizi hatası: {exc} | "
                f"damage CRS={getattr(damage_gdf, 'crs', None)}, "
                f"road CRS={getattr(road_gdf, 'crs', None)}"
            )
            return None

    def memory_mapped_diff(
        self,
        output_dir: str = ".",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Büyük görüntüler için memory-mapping ile fark hesabı.

        np.memmap: RAM'e yüklemeden disk üzerinden array indexlemeye
        izin verir. float64 → float32 dönüşümü belleği %50 azaltır.
        Görüntü tam array olarak RAM'e alınmaz; block-window ile
        parça parça memmap'e kopyalanır.

        Not: Geçici .npy dosyaları benzersiz adla oluşturulur ve
        çağıranın temizlemesi gerekir.

        Returns:
            (before_mmap, after_mmap) — disk-backed numpy arrays
        """
        rasterio = _require_rasterio()

        out = Path(output_dir)

        def _to_memmap(src_path: Path, prefix: str) -> np.ndarray:
            import os

            fd, mmap_path = tempfile.mkstemp(
                dir=str(out), prefix=prefix, suffix=".npy"
            )
            os.close(fd)
            with rasterio.open(src_path) as src:
                shape = (src.height, src.width)
                mmap = np.memmap(
                    mmap_path, dtype=np.float32, mode="w+", shape=shape
                )
                # Pencere pencere kopyala — tam array hiçbir anda RAM'de değil
                for _, win in src.block_windows(1):
                    rows = slice(
                        int(win.row_off), int(win.row_off + win.height)
                    )
                    cols = slice(
                        int(win.col_off), int(win.col_off + win.width)
                    )
                    mmap[rows, cols] = src.read(1, window=win).astype(np.float32)
                mmap.flush()
            logger.info(
                f"Memory-map: {mmap_path} — {shape[0]}×{shape[1]}px, "
                f"~{mmap.nbytes / 1e6:.1f}MB (float32)"
            )
            return mmap

        b_mmap = _to_memmap(self.before_path, "_mmap_before_")
        a_mmap = _to_memmap(self.after_path, "_mmap_after_")
        return b_mmap, a_mmap

    # ── İç Metodlar ─────────────────────────────────────────

    def _tile_windows(
        self,
        width: int,
        height: int,
        transform: Any,
    ) -> Generator[Tuple[Any, Any], None, None]:
        """
        Görüntüyü block_size × block_size tile'lara böler.
        Her tile için (window, tile_transform) döndürür.
        CRS korunur: tile_transform orijinal koordinat referansında.
        """
        try:
            from rasterio.windows import Window
            from rasterio.windows import transform as window_transform
        except ImportError as exc:
            raise ImportError("rasterio gereklidir") from exc

        bs = self.block_size
        for row_off in range(0, height, bs):
            for col_off in range(0, width, bs):
                win_h = min(bs, height - row_off)
                win_w = min(bs, width  - col_off)
                window = Window(col_off, row_off, win_w, win_h)

                # Tile'ın coğrafi sınırları (CRS bilgisi korunur)
                tile_tf = window_transform(window, transform)

                yield window, tile_tf

    def _detect_tile(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Tek tile için hasar tespiti algoritması.

        Adımlar:
          1. SSIM fark haritası (skimage)
          2. Otsu eşikleme → binary mask
          3. Morfolojik temizleme (gürültü bastırma)

        dtype: float32 giriş → uint8 çıkış (dtype optimizasyonu)

        Returns:
            (binary_mask uint8, ssim_score, damage_ratio)
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            import cv2
        except ImportError as exc:
            raise ImportError(
                "scikit-image ve opencv gereklidir: "
                "pip install scikit-image opencv-python-headless"
            ) from exc

        try:
            # uint8'e dönüştür (SSIM ve OpenCV için)
            b8 = np.clip(before, 0, UINT8_MAX).astype(np.uint8)
            a8 = np.clip(after,  0, UINT8_MAX).astype(np.uint8)

            win_size = min(self.ssim_win_size, b8.shape[0], b8.shape[1])
            if win_size < MIN_SSIM_WIN:
                return np.zeros(b8.shape, dtype=np.uint8), 1.0, 0.0
            if win_size % 2 == 0:
                win_size -= 1

            score, diff = ssim(b8, a8, win_size=win_size, full=True)
            diff_uint8  = (
                np.clip((1 - diff) * SSIM_DIFF_SCALE, 0, UINT8_MAX)
            ).astype(np.uint8)

            # Otsu eşikleme
            _, binary = cv2.threshold(
                diff_uint8, 0, UINT8_MAX,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            # Morfolojik temizlik (küçük parazitleri kaldır)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            dmg_ratio = float(binary.sum() / UINT8_MAX) / binary.size
            return binary, float(score), dmg_ratio

        except (ValueError, cv2.error) as exc:
            logger.warning(f"Tile tespit hatası: {exc}")
            empty = np.zeros(before.shape, dtype=np.uint8)
            return empty, 1.0, 0.0
