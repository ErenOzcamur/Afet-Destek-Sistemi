# ============================================================
# analyzer.py - ASTRO-RESQ Analiz Motoru
# TUA Astro Hackathon 2026
# ============================================================

"""
Modüller:
  DamageEngine       - SSIM + Morfoloji tabanlı yapısal hasar tespiti
  RouteAnalyzer      - OSMnx tabanlı yol erişilebilirlik ve güvenli rota
  NotificationEngine - Analiz sonuçlarından kritik uyarı üreticisi

Yerli Uydu Çözünürlüğü:
  İMECE    : 0.9 m/px  → 1 px = 0.81 m²
  GÖKTÜRK-1: 2.5 m/px  → 1 px = 6.25 m²
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import cv2
import numpy as np
from loguru import logger
from skimage.metrics import structural_similarity as ssim

from config import cv_config, routing_config, CVConfig

# ── Modül sabitleri ─────────────────────────────────────────
# SSIM/diff birleştirme ağırlıkları (detect_changes + run_full_analysis)
SSIM_DAMAGE_WEIGHT = 0.6
DIFF_WEIGHT = 0.4
# Heatmap harmanlama ağırlıkları
HEATMAP_BASE_ALPHA = 0.45
HEATMAP_OVERLAY_ALPHA = 0.55
# Yol engelleme parametreleri
DEFAULT_TRAVEL_TIME_S = 60
BLOCKED_PENALTY_FACTOR = 100


# ============================================================
#  DATA STRUCTURES
# ============================================================

@dataclass
class DamageRegion:
    """Tespit edilen tek bir hasar bölgesi."""
    contour: np.ndarray
    area_px: float
    area_m2: float                              # area_px * resolution²
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]     # x, y, w, h
    severity: str                               # "low" | "mid" | "high"
    severity_score: float
    is_building: bool = False
    geo_centroid: Optional[Tuple[float, float]] = None


@dataclass
class BlockedRoad:
    """Hasar maskesiyle çakışan yol segmenti."""
    road_name: str
    osm_id: Optional[str]
    geometry_coords: List[Tuple[float, float]]
    overlap_ratio: float
    length_m: float


class Alert(NamedTuple):
    """Analiz sonucu üretilen kritik uyarı."""
    level: str    # "critical" | "warning" | "info"
    icon: str
    title: str
    detail: str


@dataclass
class DisasterReport:
    """Tam afet analizi raporu — engine.py ile tam uyumlu."""
    ssim_score: float
    ssim_map: np.ndarray
    diff_map: np.ndarray
    heatmap: np.ndarray
    binary_mask: np.ndarray
    regions: List[DamageRegion]
    destroyed_buildings: int = 0
    heavy_damage_buildings: int = 0
    moderate_damage_buildings: int = 0
    blocked_roads: List[BlockedRoad] = field(default_factory=list)
    total_blocked_length_m: float = 0.0
    summary: Dict = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)

    # ── Properties ──────────────────────────────────────────

    @property
    def total_damage_area_px(self) -> float:
        """Tüm bölgelerin piksel cinsinden toplam alanı."""
        return sum(r.area_px for r in self.regions)

    @property
    def total_damage_area_m2(self) -> float:
        """Tüm bölgelerin m² cinsinden toplam alanı."""
        return sum(r.area_m2 for r in self.regions)

    # ── Methods ─────────────────────────────────────────────

    def get_executive_summary(self) -> Dict:
        """Yönetici özeti sözlüğü döndürür."""
        severity_counts: Dict[str, int] = {"low": 0, "mid": 0, "high": 0}
        for r in self.regions:
            severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

        total_px = max(1, self.binary_mask.size)
        damaged_px = int(np.sum(self.binary_mask > 0))
        damage_ratio = damaged_px / total_px

        return {
            "ssim_score": round(self.ssim_score, 4),
            "destroyed_buildings": self.destroyed_buildings,
            "heavy_damage_buildings": self.heavy_damage_buildings,
            "moderate_damage_buildings": self.moderate_damage_buildings,
            "blocked_roads": len(self.blocked_roads),
            "blocked_road_length_m": round(self.total_blocked_length_m, 1),
            "total_regions": len(self.regions),
            "damage_ratio": round(damage_ratio, 4),
            "damage_area_m2": round(self.total_damage_area_m2, 1),
            "severity_distribution": severity_counts,
        }

    def blocked_roads_to_json(self) -> str:
        """Kapanan yolları JSON string olarak döndürür."""
        data = [
            {
                "road_name": br.road_name,
                "osm_id": br.osm_id,
                "geometry_coords": br.geometry_coords,
                "overlap_ratio": round(br.overlap_ratio, 4),
                "length_m": round(br.length_m, 1),
            }
            for br in self.blocked_roads
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def blocked_roads_to_csv_rows(self) -> List[Dict]:
        """Kapanan yolları CSV satır listesi olarak döndürür."""
        return [
            {
                "road_name": br.road_name,
                "osm_id": br.osm_id or "",
                "overlap_ratio": round(br.overlap_ratio, 4),
                "length_m": round(br.length_m, 1),
                "start_lat": br.geometry_coords[0][0] if br.geometry_coords else "",
                "start_lon": br.geometry_coords[0][1] if br.geometry_coords else "",
            }
            for br in self.blocked_roads
        ]


# ============================================================
#  DAMAGE ENGINE
# ============================================================

class DamageEngine:
    """
    SSIM + Morfoloji tabanlı yapısal hasar tespit motoru.

    Parametreler
    ------------
    before_img : np.ndarray   Afet öncesi görüntü (BGR veya RGB, uint8)
    after_img  : np.ndarray   Afet sonrası görüntü (BGR veya RGB, uint8)
    config     : CVConfig     Computer vision parametreleri (None → varsayılan)
    resolution_m_per_pixel : float
        İMECE=0.9, GÖKTÜRK-1=2.5 (piksel başına metre)
    """

    def __init__(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        config: Optional[CVConfig] = None,
        resolution_m_per_pixel: float = 0.9,
    ):
        self.before_img = before_img
        self.after_img = after_img
        self.cfg = config or cv_config
        self.resolution = resolution_m_per_pixel
        self._binary_mask: Optional[np.ndarray] = None
        self._regions: List[DamageRegion] = []
        logger.info(
            f"DamageEngine başlatıldı | çözünürlük={resolution_m_per_pixel}m/px"
        )

    # ── Internal helpers ────────────────────────────────────

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """BGR/RGB görüntüyü gri tona dönüştürür."""
        if img.ndim == 2:
            return img
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _match_dimensions(
        self, a: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """b'yi a'nın boyutlarına ölçekler."""
        if a.shape != b.shape:
            logger.warning(
                f"Boyut uyumsuzluğu {a.shape} vs {b.shape}, yeniden boyutlandırılıyor."
            )
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        return a, b

    def _prepare_grays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Her iki görüntüyü gri tona çevirir ve boyutlarını eşleştirir."""
        gray_b = self._to_gray(self.before_img)
        gray_a = self._to_gray(self.after_img)
        return self._match_dimensions(gray_b, gray_a)

    def _compute_ssim(
        self, gray_b: np.ndarray, gray_a: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """SSIM skorunu ve haritasını hesaplar."""
        score, ssim_map = ssim(
            gray_b, gray_a,
            win_size=self.cfg.ssim_win_size,
            gaussian_weights=self.cfg.ssim_gaussian_weights,
            full=True,
        )
        return float(score), ssim_map

    def _severity_score_from_area(self, area_px: float) -> float:
        """Piksel alanından normalize severity skoru (0-1) üretir."""
        return min(area_px / max(1, self.cfg.building_max_area), 1.0)

    def _classify_severity(self, score: float) -> str:
        """Hasar skoru → severity etiketi."""
        if score >= self.cfg.damage_threshold_high:
            return "high"
        if score >= self.cfg.damage_threshold_mid:
            return "mid"
        return "low"

    # ── Public methods ──────────────────────────────────────

    def detect_changes(self) -> np.ndarray:
        """
        SSIM + piksel farkı tabanlı ikili hasar maskesi döndürür.
        Döndürülen değer: binary_mask (uint8, 0/255)
        """
        try:
            gray_b, gray_a = self._prepare_grays()
            _, ssim_map = self._compute_ssim(gray_b, gray_a)
            ssim_damage = (1.0 - ssim_map).astype(np.float32)

            diff = cv2.absdiff(gray_b, gray_a).astype(np.float32) / 255.0

            combined = SSIM_DAMAGE_WEIGHT * ssim_damage + DIFF_WEIGHT * diff
            combined_u8 = (combined * 255).astype(np.uint8)

            _, binary = cv2.threshold(
                combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, self.cfg.morph_kernel_size
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            self._binary_mask = binary
            logger.info("detect_changes tamamlandı.")
            return binary

        except Exception as exc:
            logger.error(f"detect_changes hatası: {exc}")
            h = self.before_img.shape[0]
            w = self.before_img.shape[1]
            return np.zeros((h, w), dtype=np.uint8)

    def estimate_damage(self) -> Dict:
        """
        Hasar istatistiklerini hesaplar.

        Döndürür
        --------
        dict içinde: destroyed_buildings, heavy_damage_buildings,
        moderate_damage_buildings, damage_area_m2, damage_ratio,
        ssim_score, total_regions, blocked_roads, blocked_road_length_m
        """
        try:
            if self._binary_mask is None:
                self.detect_changes()

            contours = self.get_damage_contours()
            destroyed = heavy = moderate = 0
            total_m2 = 0.0
            res2 = self.resolution ** 2

            for cnt in contours:
                area_px = float(cv2.contourArea(cnt))
                if area_px < self.cfg.min_contour_area:
                    continue
                area_m2 = area_px * res2
                total_m2 += area_m2

                # severity score: normalise based on building max area
                score = min(area_px / self.cfg.building_max_area, 1.0)
                sev = self._classify_severity(score)
                if sev == "high":
                    destroyed += 1
                elif sev == "mid":
                    heavy += 1
                else:
                    moderate += 1

            gray_b = self._to_gray(self.before_img)
            gray_a = self._to_gray(self.after_img)
            gray_b, gray_a = self._match_dimensions(gray_b, gray_a)
            score_val, _ = ssim(
                gray_b, gray_a,
                win_size=self.cfg.ssim_win_size,
                gaussian_weights=self.cfg.ssim_gaussian_weights,
                full=True,
            )

            total_px = int(self._binary_mask.size)
            damaged_px = int(np.sum(self._binary_mask > 0))
            damage_ratio = damaged_px / max(1, total_px)

            return {
                "destroyed_buildings": destroyed,
                "heavy_damage_buildings": heavy,
                "moderate_damage_buildings": moderate,
                "damage_area_m2": round(total_m2, 1),
                "damage_ratio": round(damage_ratio, 4),
                "ssim_score": round(float(score_val), 4),
                "total_regions": len(contours),
                "blocked_roads": 0,
                "blocked_road_length_m": 0.0,
            }
        except Exception as exc:
            logger.error(f"estimate_damage hatası: {exc}")
            return {
                "destroyed_buildings": 0, "heavy_damage_buildings": 0,
                "moderate_damage_buildings": 0, "damage_area_m2": 0.0,
                "damage_ratio": 0.0, "ssim_score": 1.0,
                "total_regions": 0, "blocked_roads": 0,
                "blocked_road_length_m": 0.0,
            }

    def get_damage_contours(self) -> List:
        """Hasar maskesindeki OpenCV konturlarını döndürür."""
        try:
            if self._binary_mask is None:
                self.detect_changes()
            contours, _ = cv2.findContours(
                self._binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filtered = [
                c for c in contours
                if cv2.contourArea(c) >= self.cfg.min_contour_area
            ]
            logger.info(f"Kontur sayısı: {len(filtered)}")
            return filtered
        except Exception as exc:
            logger.error(f"get_damage_contours hatası: {exc}")
            return []

    def run_full_analysis(
        self, geo_transform=None
    ) -> "DisasterReport":
        """
        Tam analiz pipeline'ı çalıştırır.

        Adımlar
        -------
        1. Gri ton + boyut eşleştirme
        2. SSIM skoru + ssim_map
        3. absdiff → diff_map
        4. Ağırlıklı birleştirme → Otsu eşikleme → morfolojik işlem → binary_mask
        5. findContours → DamageRegion nesneleri
        6. Bina sınıflandırma (high→destroyed, mid→heavy, low→moderate)
        7. Geo koordinat atama (transform varsa)
        8. JET colormap heatmap
        """
        logger.info("Tam analiz başlatıldı.")
        try:
            # 1. Gri ton + boyut eşleştirme
            gray_b = self._to_gray(self.before_img)
            gray_a = self._to_gray(self.after_img)
            gray_b, gray_a = self._match_dimensions(gray_b, gray_a)

            # Görüntüleri kaydet (after_img boyutunu hizala)
            after_display = self.after_img
            if after_display.shape[:2] != gray_b.shape[:2]:
                after_display = cv2.resize(
                    after_display, (gray_b.shape[1], gray_b.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

            # 2. SSIM
            win = self.cfg.ssim_win_size
            ssim_val, ssim_map_raw = ssim(
                gray_b, gray_a,
                win_size=win,
                gaussian_weights=self.cfg.ssim_gaussian_weights,
                full=True,
            )
            ssim_score = float(ssim_val)
            ssim_damage = (1.0 - ssim_map_raw).astype(np.float32)
            ssim_map_vis = (ssim_map_raw * 255).clip(0, 255).astype(np.uint8)

            # 3. absdiff → diff_map
            diff_raw = cv2.absdiff(gray_b, gray_a).astype(np.float32) / 255.0
            diff_map_vis = (diff_raw * 255).clip(0, 255).astype(np.uint8)

            # 4. Birleştir → Otsu → morfoloji → binary_mask
            combined = (0.6 * ssim_damage + 0.4 * diff_raw)
            combined_u8 = (combined * 255).clip(0, 255).astype(np.uint8)
            _, binary = cv2.threshold(
                combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, self.cfg.morph_kernel_size
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self._binary_mask = binary

            # 5. findContours → DamageRegion
            contours_raw, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            res2 = self.resolution ** 2
            regions: List[DamageRegion] = []

            for cnt in contours_raw:
                area_px = float(cv2.contourArea(cnt))
                if area_px < self.cfg.min_contour_area:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                x, y, w, h = cv2.boundingRect(cnt)

                area_m2 = area_px * res2
                severity_score = min(area_px / max(1, self.cfg.building_max_area), 1.0)
                severity = self._classify_severity(severity_score)

                # 6. Bina sınıflandırması
                is_building = (
                    self.cfg.building_min_area <= area_px <= self.cfg.building_max_area
                )

                # 7. Geo koordinat
                geo_centroid: Optional[Tuple[float, float]] = None
                if geo_transform is not None:
                    try:
                        # rasterio Affine transform: (col, row) → (lon, lat)
                        lon = geo_transform.c + cx * geo_transform.a + cy * geo_transform.b
                        lat = geo_transform.f + cx * geo_transform.d + cy * geo_transform.e
                        geo_centroid = (float(lat), float(lon))
                    except Exception as te:
                        logger.warning(f"Geo dönüşüm hatası: {te}")

                regions.append(DamageRegion(
                    contour=cnt,
                    area_px=area_px,
                    area_m2=area_m2,
                    centroid=(cx, cy),
                    bounding_box=(x, y, w, h),
                    severity=severity,
                    severity_score=severity_score,
                    is_building=is_building,
                    geo_centroid=geo_centroid,
                ))

            # Bina sayılarını hesapla
            destroyed = sum(
                1 for r in regions if r.is_building and r.severity == "high"
            )
            heavy = sum(
                1 for r in regions if r.is_building and r.severity == "mid"
            )
            moderate = sum(
                1 for r in regions if r.is_building and r.severity == "low"
            )

            # 8. JET heatmap
            heatmap = cv2.applyColorMap(combined_u8, cv2.COLORMAP_JET)
            if after_display.ndim == 2:
                after_bgr = cv2.cvtColor(after_display, cv2.COLOR_GRAY2BGR)
            elif after_display.shape[2] == 4:
                after_bgr = cv2.cvtColor(after_display, cv2.COLOR_RGBA2BGR)
            else:
                after_bgr = cv2.cvtColor(after_display, cv2.COLOR_RGB2BGR)
            heatmap = cv2.addWeighted(after_bgr, 0.45, heatmap, 0.55, 0)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            logger.info(
                f"Analiz tamamlandı | SSIM={ssim_score:.4f} | bölge={len(regions)} "
                f"| yıkık={destroyed} | ağır={heavy} | orta={moderate}"
            )

            return DisasterReport(
                ssim_score=ssim_score,
                ssim_map=ssim_map_vis,
                diff_map=diff_map_vis,
                heatmap=heatmap,
                binary_mask=binary,
                regions=regions,
                destroyed_buildings=destroyed,
                heavy_damage_buildings=heavy,
                moderate_damage_buildings=moderate,
            )

        except Exception as exc:
            logger.error(f"run_full_analysis kritik hata: {exc}")
            h, w = self.before_img.shape[:2]
            empty = np.zeros((h, w), dtype=np.uint8)
            return DisasterReport(
                ssim_score=0.0,
                ssim_map=empty,
                diff_map=empty,
                heatmap=np.zeros((h, w, 3), dtype=np.uint8),
                binary_mask=empty,
                regions=[],
            )


# ============================================================
#  ROUTE ANALYZER
# ============================================================

class RouteAnalyzer:
    """
    OSMnx tabanlı yol erişilebilirlik ve güvenli rota analizi.

    Parametreler
    ------------
    center   : (lat, lon)   Analiz merkezi
    radius_m : float        Metre cinsinden analiz yarıçapı
    config   : RoutingConfig (None → varsayılan)
    """

    def __init__(
        self,
        center: Tuple[float, float],
        radius_m: float = 3000,
        config=None,
    ):
        self.center = center
        self.radius_m = radius_m
        self.cfg = config or routing_config
        self.graph = None
        self._ox = None
        self._nx = None

        # Lazy import
        try:
            import osmnx as ox
            import networkx as nx
            self._ox = ox
            self._nx = nx
            logger.info("OSMnx ve NetworkX başarıyla import edildi.")
        except ImportError as ie:
            logger.warning(f"OSMnx/NetworkX import edilemedi: {ie}")

    def load_network(self) -> bool:
        """
        OSM'den yol ağını indirir.
        Başarı durumunda True, hata durumunda False döndürür.
        """
        if self._ox is None:
            logger.error("OSMnx mevcut değil, ağ yüklenemiyor.")
            return False
        try:
            ox = self._ox
            lat, lon = self.center
            logger.info(
                f"Yol ağı indiriliyor: center=({lat},{lon}), r={self.radius_m}m"
            )
            G = ox.graph_from_point(
                (lat, lon),
                dist=self.radius_m,
                network_type=self.cfg.network_type,
            )
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            self.graph = G
            logger.info(
                f"Yol ağı yüklendi: {G.number_of_nodes()} düğüm, "
                f"{G.number_of_edges()} kenar"
            )
            return True
        except Exception as exc:
            logger.error(f"load_network hatası: {exc}")
            return False

    def mark_blocked_roads(
        self,
        damage_coords: List[Tuple[float, float]],
        buffer_m: Optional[float] = None,
    ) -> int:
        """
        Hasar koordinatları yakınındaki yolları engelli olarak işaretler.

        Parametreler
        ------------
        damage_coords : [(lat, lon), ...]
        buffer_m      : metre cinsinden tampon mesafesi

        Döndürür
        --------
        Engellenen kenar sayısı (int)
        """
        if self.graph is None:
            logger.warning("Yol ağı yüklenmemiş; mark_blocked_roads atlandı.")
            return 0
        try:
            ox = self._ox
            nx = self._nx
            buf = buffer_m or self.cfg.damage_buffer_radius
            blocked_count = 0

            for lat, lon in damage_coords:
                try:
                    center_node = ox.distance.nearest_nodes(self.graph, lon, lat)
                    subgraph = nx.ego_graph(
                        self.graph, center_node,
                        radius=buf, distance="length",
                    )
                    for u, v, k, data in self.graph.edges(keys=True, data=True):
                        if u in subgraph.nodes or v in subgraph.nodes:
                            data["travel_time"] = data.get("travel_time", 60) * 100
                            data["damaged"] = True
                            blocked_count += 1
                except Exception as ne:
                    logger.warning(f"Düğüm işaretleme hatası ({lat},{lon}): {ne}")

            logger.info(f"Engellenen kenar sayısı: {blocked_count}")
            return blocked_count
        except Exception as exc:
            logger.error(f"mark_blocked_roads hatası: {exc}")
            return 0

    def find_safe_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> Optional[Dict]:
        """
        Dijkstra ile güvenli rota bulur.

        Döndürür
        --------
        {'coordinates', 'total_length_m', 'total_time_s', 'passes_damage'}
        veya None (rota bulunamazsa)
        """
        if self.graph is None:
            logger.warning("Yol ağı yüklenmemiş.")
            return None
        try:
            ox = self._ox
            nx = self._nx

            orig_node = ox.distance.nearest_nodes(
                self.graph, origin[1], origin[0]
            )
            dest_node = ox.distance.nearest_nodes(
                self.graph, destination[1], destination[0]
            )

            path = nx.shortest_path(
                self.graph, orig_node, dest_node,
                weight=self.cfg.weight_attribute,
            )

            coords = []
            total_length = 0.0
            total_time = 0.0
            passes_damage = False

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.graph[u][v]
                # Çoklu kenarlarda ilkini al
                if isinstance(edge_data, dict) and 0 in edge_data:
                    edge_data = edge_data[0]
                total_length += edge_data.get("length", 0)
                total_time += edge_data.get("travel_time", 0)
                if edge_data.get("damaged", False):
                    passes_damage = True

            for node in path:
                node_data = self.graph.nodes[node]
                coords.append((node_data["y"], node_data["x"]))

            logger.info(
                f"Güvenli rota: {len(coords)} nokta, "
                f"{total_length:.0f}m, {total_time:.0f}s"
            )
            return {
                "coordinates": coords,
                "total_length_m": round(total_length, 1),
                "total_time_s": round(total_time, 1),
                "passes_damage": passes_damage,
            }
        except Exception as exc:
            logger.error(f"find_safe_route hatası: {exc}")
            return None

    def get_blocked_roads(self) -> List[BlockedRoad]:
        """
        Engellenen yol segmentlerini BlockedRoad listesi olarak döndürür.
        """
        if self.graph is None:
            return []
        try:
            blocked: List[BlockedRoad] = []
            seen_edges: Set[str] = set()

            for u, v, k, data in self.graph.edges(keys=True, data=True):
                if not data.get("damaged", False):
                    continue
                edge_key = f"{u}-{v}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                road_name = data.get("name", "Bilinmeyen Yol")
                if isinstance(road_name, list):
                    road_name = road_name[0]

                osm_id = str(data.get("osmid", "")) or None

                # Geometri
                geom_coords: List[Tuple[float, float]] = []
                if "geometry" in data:
                    try:
                        geom_coords = list(data["geometry"].coords)
                        geom_coords = [(lat, lon) for lon, lat in geom_coords]
                    except Exception:
                        pass
                if not geom_coords:
                    u_data = self.graph.nodes[u]
                    v_data = self.graph.nodes[v]
                    geom_coords = [
                        (u_data["y"], u_data["x"]),
                        (v_data["y"], v_data["x"]),
                    ]

                length_m = float(data.get("length", 0.0))
                overlap = min(1.0, length_m / max(1, self.radius_m))

                blocked.append(BlockedRoad(
                    road_name=road_name,
                    osm_id=osm_id,
                    geometry_coords=geom_coords,
                    overlap_ratio=round(overlap, 4),
                    length_m=round(length_m, 1),
                ))

            logger.info(f"Engellenen yol sayısı: {len(blocked)}")
            return blocked
        except Exception as exc:
            logger.error(f"get_blocked_roads hatası: {exc}")
            return []


# ============================================================
#  NOTIFICATION ENGINE
# ============================================================

class NotificationEngine:
    """
    Analiz raporundan kritik uyarı üreticisi.
    Tüm metodlar statiktir.
    """

    # Eşik değerleri
    DESTROY_RATIO_CRITICAL: float = 0.15
    DESTROY_RATIO_WARNING: float = 0.05
    BLOCKED_LEN_CRITICAL: float = 1000.0
    BLOCKED_ROADS_WARNING: int = 3
    DESTROYED_BUILDINGS_CRITICAL: int = 3

    @staticmethod
    def generate(report: "DisasterReport") -> List[Alert]:
        """
        Raporu analiz ederek kritik uyarı listesi döndürür.

        Kural sırası
        ------------
        1. Hasar oranı ≥ 0.15 → CRITICAL yapısal yıkım
        2. Hasar oranı ≥ 0.05 → WARNING orta düzey hasar
        3. Yıkık bina ≥ 3     → CRITICAL bina yıkımı
        4. Kapanan yol ≥ 3    → WARNING yol kapatma
        5. SSIM < 0.65        → CRITICAL altyapı erişim riski
        6. 0.65 ≤ SSIM < 0.80 → WARNING orta düzey değişim
        7. Hiç uyarı yok      → INFO temiz
        """
        alerts: List[Alert] = []
        try:
            es = report.get_executive_summary()
            damage_ratio: float = es.get("damage_ratio", 0.0)
            destroyed: int = es.get("destroyed_buildings", 0)
            blocked_count: int = es.get("blocked_roads", 0)
            blocked_len: float = es.get("blocked_road_length_m", 0.0)
            ssim_score: float = es.get("ssim_score", 1.0)
            damage_m2: float = es.get("damage_area_m2", 0.0)

            NC = NotificationEngine

            # 1. Yüksek hasar oranı → CRITICAL
            if damage_ratio >= NC.DESTROY_RATIO_CRITICAL:
                pct = round(damage_ratio * 100, 1)
                alerts.append(Alert(
                    level="critical",
                    icon="🔴",
                    title=f"%{pct} Yapısal Yıkım Tespit Edildi",
                    detail=(
                        f"Toplam hasar alanı: {damage_m2:.0f} m² | "
                        f"Acil müdahale koordinasyonu gerekiyor."
                    ),
                ))

            # 2. Orta hasar oranı → WARNING
            elif damage_ratio >= NC.DESTROY_RATIO_WARNING:
                pct = round(damage_ratio * 100, 1)
                alerts.append(Alert(
                    level="warning",
                    icon="🟠",
                    title=f"%{pct} Orta Düzey Hasar",
                    detail=(
                        f"Hasar alanı: {damage_m2:.0f} m² | "
                        f"Bölge incelemesi önerilir."
                    ),
                ))

            # 3. Çok sayıda yıkık bina → CRITICAL
            if destroyed >= NC.DESTROYED_BUILDINGS_CRITICAL:
                alerts.append(Alert(
                    level="critical",
                    icon="🏚️",
                    title=f"{destroyed} Bina Tamamen Yıkıldı",
                    detail=(
                        "Arama-kurtarma ekipleri derhal görevlendirilmeli. "
                        "Enkaz altında hayatta kalan olabilir."
                    ),
                ))

            # 4. Kapanan yollar → WARNING
            if blocked_count >= NC.BLOCKED_ROADS_WARNING:
                km = round(blocked_len / 1000, 2)
                alerts.append(Alert(
                    level="warning",
                    icon="🚧",
                    title=f"{blocked_count} Yol Segmenti Kapalı",
                    detail=(
                        f"Toplam kapanan yol uzunluğu: {km} km | "
                        f"Alternatif güzergâhlar kullanılmalı."
                    ),
                ))

            # 5. Çok düşük SSIM → CRITICAL altyapı
            if ssim_score < 0.65:
                alerts.append(Alert(
                    level="critical",
                    icon="🏥",
                    title="Kritik Altyapı Erişimi Risk Altında",
                    detail=(
                        f"SSIM skoru: {ssim_score:.3f} — Hastane ve acil servis "
                        f"erişimi ciddi ölçüde bozulmuş olabilir."
                    ),
                ))

            # 6. Orta SSIM → WARNING
            elif 0.65 <= ssim_score < 0.80:
                alerts.append(Alert(
                    level="warning",
                    icon="📡",
                    title="Orta Düzey Yapısal Değişim",
                    detail=(
                        f"SSIM skoru: {ssim_score:.3f} — "
                        f"Yapısal değişiklikler gözlemleniyor, saha doğrulaması önerilir."
                    ),
                ))

            # 7. Temiz
            if not alerts:
                alerts.append(Alert(
                    level="info",
                    icon="✅",
                    title="Kritik Hasar Tespit Edilmedi",
                    detail=(
                        f"SSIM: {ssim_score:.3f} | "
                        f"Hasar oranı: {damage_ratio*100:.2f}% — "
                        f"Rutin izleme yeterli."
                    ),
                ))

            logger.info(
                f"NotificationEngine: {len(alerts)} uyarı üretildi "
                f"(critical={sum(1 for a in alerts if a.level=='critical')}, "
                f"warning={sum(1 for a in alerts if a.level=='warning')})"
            )

        except Exception as exc:
            logger.error(f"NotificationEngine.generate hatası: {exc}")
            alerts.append(Alert(
                level="warning",
                icon="⚠️",
                title="Uyarı Üretilemedi",
                detail=f"Hata: {exc}",
            ))

        return alerts
