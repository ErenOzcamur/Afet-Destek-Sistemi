# ============================================================
# simulator.py - Synthetic Disaster Simulation Framework
# TUA Astro Hackathon DSS — İnternet Bağımsız Test Altyapısı
# ============================================================

"""
SimulationEngine: 1024×1024 sentetik afet verisi üretir.

Amaç:
  - Planet SkySat (0.5m/px) çözünürlüğünü simüle et
  - Analyzer.py / MLRAEngine'i canlı veri olmadan kalibre et
  - Hackathon süresince API anahtarı bekleme sürelerini ortadan kaldır

Teknik doğruluk:
  - Pre-disaster : Gauss gürültülü homojen asfalt + düzgün geometrik binalar
  - Post-disaster: Morfolojik enkaz patlaması → SSIM'in hasar olarak algılayacağı
    piksel yapısı (yüksek frekans bileşeni, Canny kenar tespiti compatible)

Ground Truth sözlüğü sayesinde algorithm calibration:
  gt["blocked_edges"] → doğru bloke yol listesi
  gt["damage_ratio"]  → beklenilen hasar oranı

Bağımlılıklar: numpy, networkx, (opsiyonel) rasterio
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger


# ── Sabitler ────────────────────────────────────────────────

IMG_SIZE   = 1024          # piksel
RESOLUTION = 0.5           # m/px — Planet SkySat
BLOCK_SIZE = 32            # 1 şehir bloğu = 32px × 0.5m = 16m

# Renk değerleri (gri tonlama, 0–255)
GRAY_ASPHALT  = 90         # Asfalt yüzey
GRAY_BUILDING = 160        # Bina çatısı
GRAY_DEBRIS   = 130        # Enkaz karışık dokusu (asfalt + çöküntü)
GRAY_SKY      = 200        # Arka plan


@dataclass
class SimulationConfig:
    size: int               = IMG_SIZE
    resolution_m: float     = RESOLUTION
    block_size: int         = BLOCK_SIZE
    n_buildings: int        = 40
    n_blocked_roads: int    = 6
    damage_severity: float  = 0.4    # 0–1, enkaz yoğunluğu
    noise_std: float        = 12.0   # Gauss gürültü std
    seed: int               = 42


@dataclass
class SimulationResult:
    """Simulation motorunun çıktı paketi."""
    pre_image:    np.ndarray           # (H, W) uint8 — afet öncesi
    post_image:   np.ndarray           # (H, W) uint8 — afet sonrası
    graph:        nx.Graph             # Simüle yol ağı
    ground_truth: Dict[str, Any]       # Doğrulama verisi
    config:       SimulationConfig
    geo_transform: Optional[Any] = None   # rasterio Affine (varsa)

    def package_for_analyzer(self) -> Dict:
        """
        Analyzer.py / MLRAEngine girdi formatında veri paketi.

        Returns:
            {
              "img_before"    : np.ndarray (H,W,3) BGR,
              "img_after"     : np.ndarray (H,W,3) BGR,
              "graph"         : nx.Graph,
              "ground_truth"  : Dict,
              "resolution_m"  : float,
              "geo_transform" : rasterio.Affine | None,
            }
        """
        import cv2
        before_bgr = cv2.cvtColor(self.pre_image,  cv2.COLOR_GRAY2BGR)
        after_bgr  = cv2.cvtColor(self.post_image, cv2.COLOR_GRAY2BGR)
        return {
            "img_before":   before_bgr,
            "img_after":    after_bgr,
            "graph":        self.graph,
            "ground_truth": self.ground_truth,
            "resolution_m": self.config.resolution_m,
            "geo_transform": self.geo_transform,
        }


# ── Simulation Engine ────────────────────────────────────────

class SimulationEngine:
    """
    Sentetik afet görüntüsü + yol ağı üretici.

    Kullanım:
        engine = SimulationEngine(SimulationConfig(seed=42))
        result = engine.run()
        result.pre_image   # (1024,1024) uint8
        result.ground_truth["blocked_edges"]
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        random.seed(self.cfg.seed)

    def run(self) -> SimulationResult:
        """Tam simülasyonu çalıştırır."""
        logger.info(
            f"Simülasyon başlatıldı: {self.cfg.size}px, "
            f"{self.cfg.resolution_m}m/px, seed={self.cfg.seed}"
        )

        pre, building_rects, road_lines = self._build_pre_disaster()
        graph, road_segments, edge_map  = self._build_road_graph(road_lines)
        post, blocked_edges, gt         = self._build_post_disaster(
            pre, building_rects, road_lines, road_segments, edge_map
        )

        # Opsiyonel: rasterio Affine (sol üst köşe = 37.0°E, 38.0°N)
        geo_transform = self._make_affine()

        result = SimulationResult(
            pre_image    = pre,
            post_image   = post,
            graph        = graph,
            ground_truth = gt,
            config       = self.cfg,
            geo_transform= geo_transform,
        )
        logger.info(
            f"Simülasyon tamamlandı: "
            f"{gt['blocked_road_count']} yol bloke, "
            f"beklenen hasar oranı ~{gt['expected_damage_ratio']:.2%}"
        )
        return result

    # ── Pre-Disaster ─────────────────────────────────────────

    def _build_pre_disaster(
        self,
    ) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """
        Temiz şehir dokusu üretir.

        Teknik not: Asfalt piksel değeri ~90, Gauss gürültüsü std=12
        → SSIM hesabında "baseline" referans alınır.
        Binalar ~160, yollar ~90 → Canny edge tespiti bina kenarlarında
        yüksek gradient verir.
        """
        size = self.cfg.size
        img  = np.full((size, size), GRAY_SKY, dtype=np.uint8)

        # 1. Zemin: asfalt + düşük frekanslı homojen gürültü
        noise = self._rng.normal(0, self.cfg.noise_std * 0.5, (size, size))
        base  = np.clip(GRAY_ASPHALT + noise, 60, 120).astype(np.uint8)
        img[:] = base

        # 2. Yol gridini çiz
        road_lines = self._draw_road_grid(img)

        # 3. Binaları yerleştir
        building_rects = self._draw_buildings(img, road_lines)

        logger.debug(f"Pre-disaster: {len(building_rects)} bina, {len(road_lines)} yol")
        return img, building_rects, road_lines

    def _draw_road_grid(self, img: np.ndarray) -> List[Tuple]:
        """
        Düzenli yol grid'i çizer. 0.5m/px'de 6m yol = 12px genişlik.
        SSIM için: yollar düzgün, enkaz sonrası fark belirgin.
        """
        size = self.cfg.size
        bs   = self.cfg.block_size   # blok aralığı (piksel)
        road_w = 6                   # piksel (~3m, binek araç yolu)
        road_lines = []

        # Yatay yollar
        y = bs
        while y < size - bs:
            img[y:y + road_w, :] = np.clip(
                GRAY_ASPHALT + self._rng.normal(0, 8, (road_w, size)), 60, 115
            ).astype(np.uint8)
            road_lines.append(("H", y, y + road_w, 0, size))
            y += bs

        # Dikey yollar
        x = bs
        while x < size - bs:
            img[:, x:x + road_w] = np.clip(
                GRAY_ASPHALT + self._rng.normal(0, 8, (size, road_w)), 60, 115
            ).astype(np.uint8)
            road_lines.append(("V", x, x + road_w, 0, size))
            x += bs

        return road_lines

    def _draw_buildings(
        self, img: np.ndarray, road_lines: List[Tuple]
    ) -> List[Tuple]:
        """
        Bloklara düzgün yerleştirilmiş dikdörtgen binalar.
        Bina gri tonu ~160: SSIM hasar tespitinde kritik referans yapı.
        """
        size = self.cfg.size
        bs   = self.cfg.block_size
        road_w = 6
        rects = []

        h_roads = [r for r in road_lines if r[0] == "H"]
        v_roads = [r for r in road_lines if r[0] == "V"]

        for hi, hr in enumerate(h_roads[:-1]):
            for vi, vr in enumerate(v_roads[:-1]):
                y0 = hr[1] + road_w + 2
                y1 = h_roads[hi + 1][1] - 2 if hi + 1 < len(h_roads) else hr[1] + bs - 2
                x0 = vr[1] + road_w + 2
                x1 = v_roads[vi + 1][1] - 2 if vi + 1 < len(v_roads) else vr[1] + bs - 2

                if y1 - y0 < 6 or x1 - x0 < 6:
                    continue
                if len(rects) >= self.cfg.n_buildings:
                    break

                # Bina iç dolgusu + çatı gürültüsü
                patch = self._rng.normal(GRAY_BUILDING, 10, (y1 - y0, x1 - x0))
                img[y0:y1, x0:x1] = np.clip(patch, 130, 220).astype(np.uint8)
                rects.append((x0, y0, x1, y1))

        return rects

    # ── Road Graph ───────────────────────────────────────────

    def _build_road_graph(
        self, road_lines: List[Tuple]
    ) -> Tuple[nx.Graph, List[Dict], Dict]:
        """
        Yol çizgilerinden NetworkX graph oluşturur.
        Her kavşak → düğüm, her segment → kenar.
        Düğüm koordinatları gerçek lat/lon'a çevrilir.
        """
        size  = self.cfg.size
        bs    = self.cfg.block_size
        res   = self.cfg.resolution_m

        # Yol koordinatları (piksel → lat/lon)
        origin_lat, origin_lon = 38.0, 37.0   # Sol üst köşe
        def px_to_latlon(px, py):
            lat = origin_lat - (py * res / 111320)
            lon = origin_lon + (px * res / (111320 * math.cos(math.radians(origin_lat))))
            return lat, lon

        G = nx.Graph()
        segments = []
        edge_map: Dict[Tuple, int] = {}

        h_roads = sorted([r for r in road_lines if r[0] == "H"], key=lambda r: r[1])
        v_roads = sorted([r for r in road_lines if r[0] == "V"], key=lambda r: r[1])

        # Kavşak düğümleri
        node_id = 0
        node_grid: Dict[Tuple[int, int], int] = {}
        for hr in h_roads:
            for vr in v_roads:
                ry = (hr[1] + hr[2]) // 2
                rx = (vr[1] + vr[2]) // 2
                lat, lon = px_to_latlon(rx, ry)
                G.add_node(node_id, y=lat, x=lon, px=rx, py=ry)
                node_grid[(hr[1], vr[1])] = node_id
                node_id += 1

        # Segment kenarları
        seg_id = 0
        for hi in range(len(h_roads) - 1):
            for vi in range(len(v_roads)):
                n1 = node_grid.get((h_roads[hi][1], v_roads[vi][1]))
                n2 = node_grid.get((h_roads[hi + 1][1], v_roads[vi][1]))
                if n1 is not None and n2 is not None:
                    length = bs * res
                    G.add_edge(n1, n2, length=length, travel_time=length / (30 / 3.6))
                    rx = (v_roads[vi][1] + v_roads[vi][2]) // 2
                    y_mid = (h_roads[hi][1] + h_roads[hi + 1][1]) // 2
                    segments.append({
                        "id": seg_id, "nodes": (n1, n2),
                        "axis": "V", "px_center": (rx, y_mid),
                        "highway": "residential", "length": length,
                    })
                    edge_map[(n1, n2)] = seg_id
                    edge_map[(n2, n1)] = seg_id
                    seg_id += 1

        for vi in range(len(v_roads) - 1):
            for hi in range(len(h_roads)):
                n1 = node_grid.get((h_roads[hi][1], v_roads[vi][1]))
                n2 = node_grid.get((h_roads[hi][1], v_roads[vi + 1][1]))
                if n1 is not None and n2 is not None:
                    length = bs * res
                    G.add_edge(n1, n2, length=length, travel_time=length / (30 / 3.6))
                    x_mid = (v_roads[vi][1] + v_roads[vi + 1][1]) // 2
                    ry = (h_roads[hi][1] + h_roads[hi][2]) // 2
                    segments.append({
                        "id": seg_id, "nodes": (n1, n2),
                        "axis": "H", "px_center": (x_mid, ry),
                        "highway": "residential", "length": length,
                    })
                    edge_map[(n1, n2)] = seg_id
                    edge_map[(n2, n1)] = seg_id
                    seg_id += 1

        logger.debug(
            f"Road graph: {G.number_of_nodes()} düğüm, "
            f"{G.number_of_edges()} kenar, {len(segments)} segment"
        )
        return G, segments, edge_map

    # ── Post-Disaster ────────────────────────────────────────

    def _build_post_disaster(
        self,
        pre: np.ndarray,
        building_rects: List[Tuple],
        road_lines: List[Tuple],
        road_segments: List[Dict],
        edge_map: Dict,
    ) -> Tuple[np.ndarray, List[int], Dict]:
        """
        Afet sonrası görüntü üretir.

        Enkaz pikselleri:
          - Yüksek frekans gürültüsü (std ~45) → SSIM düşürür
          - Morfolojik blob: Canny edge tespitinde yüksek response
          - Varyans: ~80–120 → MLRAEngine'de HIGH_OBSTRUCTION eşiğini aşar
        """
        post = pre.copy()
        n_blocked = self.cfg.n_blocked_roads
        severity  = self.cfg.damage_severity

        # Bloke edilecek segmentleri seç
        all_seg_ids = list(range(len(road_segments)))
        blocked_ids = random.sample(
            all_seg_ids, min(n_blocked, len(all_seg_ids))
        )

        for seg_id in blocked_ids:
            seg = road_segments[seg_id]
            cx, cy = seg["px_center"]
            half   = self.cfg.block_size // 2

            # Enkaz bölgesi tanımla
            y0 = max(0,               cy - half)
            y1 = min(self.cfg.size,   cy + half)
            x0 = max(0,               cx - half)
            x1 = min(self.cfg.size,   cx + half)

            # Enkaz dokusu: yüksek frekanslı rasgele gürültü
            # → SSIM'in hasar olarak algılayacağı yapı
            debris = self._rng.normal(
                GRAY_DEBRIS,
                35 + severity * 30,    # yoğunluğa göre gürültü
                (y1 - y0, x1 - x0)
            )
            # Blob efekti: birkaç noktada piksel yoğunluğu yüksek
            n_blobs = int(severity * 8) + 2
            for _ in range(n_blobs):
                by = self._rng.integers(0, max(1, y1 - y0))
                bx = self._rng.integers(0, max(1, x1 - x0))
                br = self._rng.integers(2, 6)
                yb0 = max(0, by - br)
                yb1 = min(y1 - y0, by + br)
                xb0 = max(0, bx - br)
                xb1 = min(x1 - x0, bx + br)
                debris[yb0:yb1, xb0:xb1] += self._rng.normal(50, 15, (yb1 - yb0, xb1 - xb0))

            post[y0:y1, x0:x1] = np.clip(debris, 0, 255).astype(np.uint8)

        # Binalarda kısmi hasar
        dmg_buildings = random.sample(
            building_rects, min(len(building_rects) // 3, len(building_rects))
        )
        for (bx0, by0, bx1, by1) in dmg_buildings:
            # Köşeleri ve kenarları "dağıt"
            corner_h = (by1 - by0) // 3
            corner_w = (bx1 - bx0) // 3
            corner   = self._rng.normal(GRAY_DEBRIS, 40, (corner_h, corner_w))
            post[by0:by0 + corner_h, bx0:bx0 + corner_w] = \
                np.clip(corner, 0, 255).astype(np.uint8)

        # Genel atmosferik gürültü artışı (deprem tozu simülasyonu)
        atm = self._rng.normal(0, self.cfg.noise_std, post.shape)
        post = np.clip(post.astype(np.float32) + atm, 0, 255).astype(np.uint8)

        # Ground Truth sözlüğü
        blocked_nodes = set()
        for sid in blocked_ids:
            n1, n2 = road_segments[sid]["nodes"]
            blocked_nodes.update([n1, n2])

        total_px    = self.cfg.size ** 2
        debris_px   = (self.cfg.block_size ** 2) * len(blocked_ids)
        exp_dmg     = min(1.0, debris_px / total_px * 3)

        gt = {
            "blocked_segment_ids":  blocked_ids,
            "blocked_edge_count":   len(blocked_ids),
            "blocked_road_count":   len(blocked_ids),
            "blocked_node_ids":     list(blocked_nodes),
            "damaged_buildings":    len(dmg_buildings),
            "expected_damage_ratio": exp_dmg,
            "damage_severity":      severity,
            "resolution_m":         self.cfg.resolution_m,
            "n_total_segments":     len(road_segments),
        }
        logger.debug(
            f"Post-disaster: {len(blocked_ids)} yol bloke, "
            f"{len(dmg_buildings)} bina hasarlı"
        )
        return post, blocked_ids, gt

    # ── GeoTIFF Export ───────────────────────────────────────

    def save_geotiff(
        self,
        result: SimulationResult,
        pre_path:  str = "sim_pre.tif",
        post_path: str = "sim_post.tif",
    ) -> Tuple[str, str]:
        """
        GeoTIFF olarak kaydet (rasterio gerekli).
        CRS: EPSG:4326, orijin: 37.0°E 38.0°N.
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds

            size  = self.cfg.size
            res   = self.cfg.resolution_m
            lat_s = 38.0 - size * res / 111320
            lon_e = 37.0 + size * res / 111320
            tf    = from_bounds(37.0, lat_s, lon_e, 38.0, size, size)

            profile = {
                "driver": "GTiff", "dtype": "uint8",
                "width": size, "height": size, "count": 1,
                "crs": "EPSG:4326", "transform": tf,
            }

            for img, path in [(result.pre_image, pre_path),
                              (result.post_image, post_path)]:
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(img[np.newaxis, :, :])

            logger.info(f"GeoTIFF kaydedildi: {pre_path}, {post_path}")
            return pre_path, post_path

        except ImportError:
            logger.warning("rasterio yüklü değil — PNG olarak kaydediliyor")
            return self.save_png(result, pre_path.replace(".tif", ".png"),
                                 post_path.replace(".tif", ".png"))

    def save_png(
        self,
        result: SimulationResult,
        pre_path:  str = "sim_pre.png",
        post_path: str = "sim_post.png",
    ) -> Tuple[str, str]:
        """PNG olarak kaydet (Pillow)."""
        from PIL import Image
        Image.fromarray(result.pre_image).save(pre_path)
        Image.fromarray(result.post_image).save(post_path)
        logger.info(f"PNG kaydedildi: {pre_path}, {post_path}")
        return pre_path, post_path

    def save_ground_truth(
        self, result: SimulationResult, path: str = "sim_ground_truth.json"
    ) -> str:
        Path(path).write_text(
            json.dumps(result.ground_truth, ensure_ascii=False, indent=2)
        )
        logger.info(f"Ground truth kaydedildi: {path}")
        return path

    def _make_affine(self):
        """rasterio Affine transform (opsiyonel)."""
        try:
            from rasterio.transform import from_bounds
            size  = self.cfg.size
            res   = self.cfg.resolution_m
            lat_s = 38.0 - size * res / 111320
            lon_e = 37.0 + size * res / 111320
            return from_bounds(37.0, lat_s, lon_e, 38.0, size, size)
        except ImportError:
            return None


# ── Test Fonksiyonu ─────────────────────────────────────────

def run_integration_test(
    save_files: bool = False,
    output_dir: str = ".",
) -> Dict:
    """
    Simülasyonu analyzer.py + MLRAEngine ile uçtan uca test eder.

    Args:
        save_files : GeoTIFF / PNG çıktıları kaydet
        output_dir : Çıktı dizini

    Returns:
        {
          "simulation_ok":  bool,
          "analyzer_ok":    bool,
          "mlra_ok":        bool,
          "ssim_score":     float,
          "damage_ratio":   float,
          "n_blocked_found": int,
          "n_blocked_gt":    int,
          "ground_truth":   dict,
        }
    """
    results: Dict = {
        "simulation_ok": False,
        "analyzer_ok":   False,
        "mlra_ok":       False,
    }

    # 1. Simülasyon üret
    logger.info("=== INTEGRATION TEST BAŞLADI ===")
    engine = SimulationEngine(SimulationConfig(seed=42, n_blocked_roads=5))
    sim    = engine.run()
    results["simulation_ok"]  = True
    results["ground_truth"]   = sim.ground_truth

    if save_files:
        outdir = Path(output_dir)
        engine.save_geotiff(sim,
                            str(outdir / "sim_pre.tif"),
                            str(outdir / "sim_post.tif"))
        engine.save_ground_truth(sim, str(outdir / "sim_ground_truth.json"))

    # 2. Paket hazırla
    pkg = sim.package_for_analyzer()

    # 3. DamageEngine testi
    try:
        from analyzer import DamageEngine
        de  = DamageEngine(
            pkg["img_before"],
            pkg["img_after"],
            resolution_m_per_pixel=pkg["resolution_m"],
        )
        rpt = de.run_full_analysis(geo_transform=pkg["geo_transform"])
        results["analyzer_ok"]  = True
        results["ssim_score"]   = rpt.ssim_score
        results["damage_ratio"] = rpt.damage_ratio
        logger.info(
            f"DamageEngine: SSIM={rpt.ssim_score:.3f}, "
            f"hasar={rpt.damage_ratio:.2%}"
        )
    except Exception as exc:
        logger.error(f"DamageEngine test hatası: {exc}")
        results["analyzer_error"] = str(exc)

    # 4. MLRAEngine testi (yalnızca graph mevcutsa)
    try:
        from planet_engine import MLRAEngine, ObstructionLevel
        # Simüle grapta rasterio Affine yok → bbox interpolasyon kullanılır
        # Binary mask: post ile pre arasındaki fark
        if results.get("analyzer_ok"):
            mask = rpt.binary_mask
        else:
            import cv2
            diff = cv2.absdiff(pkg["img_before"], pkg["img_after"])
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        mlra = MLRAEngine(
            graph        = pkg["graph"],
            binary_mask  = mask,
            geo_transform= pkg["geo_transform"],
            resolution_m = pkg["resolution_m"],
        )
        segments = mlra.analyze_all_segments()
        n_blocked = sum(
            1 for s in segments.values()
            if s.obstruction_level == ObstructionLevel.HIGH_OBSTRUCTION
        )
        results["mlra_ok"]         = True
        results["n_blocked_found"] = n_blocked
        results["n_blocked_gt"]    = sim.ground_truth["blocked_road_count"]
        logger.info(
            f"MLRAEngine: {n_blocked} bloke tespit / "
            f"{sim.ground_truth['blocked_road_count']} gerçek"
        )
    except Exception as exc:
        logger.error(f"MLRAEngine test hatası: {exc}")
        results["mlra_error"] = str(exc)

    logger.info("=== INTEGRATION TEST TAMAMLANDI ===")
    return results


if __name__ == "__main__":
    res = run_integration_test(save_files=True, output_dir=".")
    print(json.dumps(
        {k: v for k, v in res.items() if k != "ground_truth"},
        indent=2, ensure_ascii=False
    ))
