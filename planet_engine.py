# ============================================================
# planet_engine.py - Micro-Logistic Road Assessment (MLRA)
# TUA Astro Hackathon DSS — Planet SkySat 0.5m/px Optimize
# ============================================================

"""
Micro-Logistic Road Assessment Engine.

0.5m/px çözünürlüğün teknik avantajı:
  - Standart uydular (2.5m/px): 1 piksel = 6.25m²
    → Araç boyutundaki engeller alt-piksel detayına düşer, yol
      "açık" olarak yanlış sınıflanır.
  - Planet SkySat (0.5m/px): 1 piksel = 0.25m²
    → Devrilmiş araç (~4m²) = 16 piksel; küçük enkaz bloku
      (~0.5m²) = 2 piksel → hassas texture analizi mümkündür.

Texture Analysis mantığı:
  - Temiz asfalt  : düşük varyans (~15-30), homojen gradient
  - Enkaz         : yüksek varyans (>65), yüksek LBP heterojenliği
  - Su baskını    : düşük varyans + düşük gradient (yansıma homojenliği)
  - Araç yığılması: orta varyans, dikdörtgensel pattern
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ── Sabitler ────────────────────────────────────────────────

class ObstructionLevel(str, Enum):
    OPEN             = "OPEN"             # Temiz, geçiş serbest
    RESTRICTED       = "RESTRICTED"       # Kısmi engel, yalnızca binek araç
    HIGH_OBSTRUCTION = "HIGH_OBSTRUCTION" # Tamamen kapalı, müdahale şart


class VehicleType(str, Enum):
    AMBULANCE       = "Ambulans"
    FIRE_TRUCK      = "İtfaiye"
    HEAVY_MACHINERY = "İş Makinesi"


# Araç tip kısıtları
VEHICLE_CONSTRAINTS: Dict[VehicleType, Dict] = {
    VehicleType.AMBULANCE: {
        "max_weight_t": 3.5, "min_width_m": 2.5,
        "can_use": ["motorway","trunk","primary","secondary","tertiary",
                    "residential","unclassified","service"],
    },
    VehicleType.FIRE_TRUCK: {
        "max_weight_t": 15.0, "min_width_m": 3.0,
        "can_use": ["motorway","trunk","primary","secondary","tertiary","residential"],
    },
    VehicleType.HEAVY_MACHINERY: {
        "max_weight_t": 30.0, "min_width_m": 3.5,
        "can_use": ["motorway","trunk","primary","secondary"],
    },
}

# OSM highway tipi → tahmini genişlik (m)
HIGHWAY_WIDTH_EST: Dict[str, float] = {
    "motorway": 7.0, "trunk": 6.0, "primary": 5.5,
    "secondary": 4.5, "tertiary": 3.5, "residential": 3.0,
    "service": 2.5, "footway": 1.5, "path": 1.2,
    "unclassified": 3.0,
}

# Araç rengi (harita görselleştirmesi)
VEHICLE_COLOR: Dict[str, str] = {
    "Ambulans": "#e63946",
    "İtfaiye": "#f77f00",
    "İş Makinesi": "#6a4c93",
}


@dataclass
class RoadSegment:
    """Tek bir OSMnx yol segmentinin MLRA sonucunu taşır."""
    edge_id: Tuple
    obstruction_level: ObstructionLevel
    texture_score: float          # 0=temiz asfalt, 1=tam engel
    pixel_damage_ratio: float     # Hasar piksel oranı (0–1)
    texture_variance: float       # Piksel varyansı
    coords: List[Tuple[float, float]]  # [(lat, lon), ...]
    street_name: str
    highway_type: str
    est_width_m: float
    passable_vehicles: List[VehicleType]
    obstruction_type: str         # debris/flooding/road_collapse/vehicle_pile/none
    centroid_latlon: Tuple[float, float]

    @property
    def alert_level(self) -> str:
        if self.obstruction_level == ObstructionLevel.HIGH_OBSTRUCTION:
            return "CRITICAL"
        if self.obstruction_level == ObstructionLevel.RESTRICTED:
            return "WARNING"
        return "INFO"

    @property
    def recommended_action(self) -> str:
        actions = {
            "debris":        "Ağır iş makinesi gönder — enkaz kaldırma ekibi",
            "flooding":      "Su tahliye pompası ve bot görevlendir",
            "road_collapse": "Geoteknik ekip çağır — geçiş tamamen kapat",
            "vehicle_pile":  "Çekici + trafik zabıtası gönder",
        }
        return actions.get(self.obstruction_type, "Alternatif rota kullan")


class MLRAEngine:
    """
    Micro-Logistic Road Assessment Engine.

    Planet SkySat 0.5m/px çözünürlüğe göre kalibre edilmiştir.
    OSMnx graph + binary hasar maskesi + opsiyonel gri görüntü alır.
    """

    # Texture eşikleri (0.5m/px için kalibre)
    TEXTURE_VAR_CLEAN  = 25.0   # Temiz asfalt üst varyans sınırı
    TEXTURE_VAR_DEBRIS = 65.0   # Enkaz alt varyans sınırı
    PIXEL_BUFFER_M     = 1.5    # Yol kenarı buffer (metre)

    def __init__(
        self,
        graph,                               # OSMnx MultiDiGraph
        binary_mask: np.ndarray,             # Hasar maskesi (0/255 veya bool)
        gray_image: Optional[np.ndarray] = None,  # Orijinal gri görüntü
        geo_transform=None,                  # rasterio Affine transform
        resolution_m: float = 0.5,           # m/px — Planet SkySat varsayılanı
    ):
        self.graph = graph
        self.binary_mask = (binary_mask > 0).astype(np.uint8)
        self.gray_image = gray_image
        self.geo_transform = geo_transform
        self.resolution_m = resolution_m
        self._h, self._w = binary_mask.shape[:2]
        self._segments: Dict[str, RoadSegment] = {}
        self._bbox: Optional[Tuple] = None
        # buffer piksel cinsinden
        self._buf_px = max(2, int(self.PIXEL_BUFFER_M / resolution_m))
        logger.info(
            f"MLRAEngine: {resolution_m}m/px, "
            f"maske={self._h}×{self._w}, buffer={self._buf_px}px"
        )

    # ── Public API ──────────────────────────────────────────

    def analyze_all_segments(
        self,
        damage_threshold: float = 0.25,
        texture_threshold: float = 0.40,
    ) -> Dict[str, RoadSegment]:
        """
        Tüm yol segmentlerini analiz eder.

        Args:
            damage_threshold : Hasar piksel oranı ≥ bu → RESTRICTED
            texture_threshold: Texture skoru ≥ bu → HIGH_OBSTRUCTION

        Returns:
            edge_key → RoadSegment eşlemesi
        """
        try:
            import osmnx as ox
            edges_gdf = ox.graph_to_gdfs(self.graph, nodes=False)
        except Exception as exc:
            logger.error(f"OSMnx GDF dönüşüm hatası: {exc}")
            return {}

        # Bbox hesapla (koordinat→piksel dönüşümü için)
        try:
            bounds = edges_gdf.total_bounds  # (minx, miny, maxx, maxy)
            self._bbox = (bounds[1], bounds[3], bounds[0], bounds[2])
        except Exception:
            self._bbox = None

        for idx, row in edges_gdf.iterrows():
            try:
                edge_key = f"{idx[0]}_{idx[1]}_{idx[2]}"
                highway   = row.get("highway", "unclassified")
                if isinstance(highway, list):
                    highway = highway[0]
                highway = str(highway)

                name = row.get("name", "İsimsiz Yol")
                if isinstance(name, list):
                    name = name[0]
                name = str(name or "İsimsiz Yol")

                est_width = HIGHWAY_WIDTH_EST.get(highway, 3.0)

                geom = row.get("geometry")
                if geom is None:
                    continue
                coords = [(lat, lon) for lon, lat in geom.coords]
                if not coords:
                    continue

                cen_lat = sum(c[0] for c in coords) / len(coords)
                cen_lon = sum(c[1] for c in coords) / len(coords)

                dmg_ratio              = self._sample_damage(coords)
                tex_score, tex_var     = self._analyze_texture(coords)
                obs_level, obs_type    = self._classify(
                    dmg_ratio, tex_score, tex_var, damage_threshold, texture_threshold
                )
                passable = self._passable_vehicles(obs_level, highway, est_width)

                self._segments[edge_key] = RoadSegment(
                    edge_id=idx,
                    obstruction_level=obs_level,
                    texture_score=tex_score,
                    pixel_damage_ratio=dmg_ratio,
                    texture_variance=tex_var,
                    coords=coords,
                    street_name=name,
                    highway_type=highway,
                    est_width_m=est_width,
                    passable_vehicles=passable,
                    obstruction_type=obs_type,
                    centroid_latlon=(cen_lat, cen_lon),
                )
            except Exception as exc:
                logger.debug(f"Segment {idx} analiz hatası: {exc}")

        n_open = sum(1 for s in self._segments.values()
                     if s.obstruction_level == ObstructionLevel.OPEN)
        n_rest = sum(1 for s in self._segments.values()
                     if s.obstruction_level == ObstructionLevel.RESTRICTED)
        n_high = sum(1 for s in self._segments.values()
                     if s.obstruction_level == ObstructionLevel.HIGH_OBSTRUCTION)
        logger.info(
            f"MLRA tamamlandı: {len(self._segments)} segment — "
            f"Açık:{n_open} Kısıtlı:{n_rest} Kapalı:{n_high}"
        )
        return self._segments

    def tactical_route(
        self,
        origin: Tuple[float, float],
        dest: Tuple[float, float],
        vehicle_type: VehicleType,
    ) -> Optional[Dict]:
        """
        Araç tipine göre taktik rota hesaplar.

        0.5m/px avantajı: dar sokak (2.5m genişlik) ile ana cadde
        ayrımı piksel hassasiyetinde yapılır; ağır makineler için
        bu fark hayat kurtarıcıdır.

        Returns:
            {coords, distance_m, travel_time_s, blocked_streets, vehicle, color}
        """
        try:
            import osmnx as ox
            import networkx as nx

            constraints  = VEHICLE_CONSTRAINTS[vehicle_type]
            allowed_hway = set(constraints["can_use"])
            min_width    = constraints["min_width_m"]
            G            = self.graph.copy()

            for u, v, k, data in G.edges(keys=True, data=True):
                hw = data.get("highway", "unclassified")
                if isinstance(hw, list):
                    hw = hw[0]
                hw = str(hw)

                edge_key = f"{u}_{v}_{k}"
                seg      = self._segments.get(edge_key)
                base_w   = data.get("travel_time", data.get("length", 100))

                # Yol tipi kısıtı
                if hw not in allowed_hway:
                    base_w *= 10

                # Genişlik kısıtı
                if HIGHWAY_WIDTH_EST.get(hw, 3.0) < min_width:
                    base_w *= 5

                # Hasar ağırlığı
                if seg:
                    if seg.obstruction_level == ObstructionLevel.HIGH_OBSTRUCTION:
                        base_w = 9_999_999
                    elif seg.obstruction_level == ObstructionLevel.RESTRICTED:
                        if vehicle_type in (VehicleType.FIRE_TRUCK,
                                            VehicleType.HEAVY_MACHINERY):
                            base_w *= 8
                        else:
                            base_w *= 2

                G[u][v][k]["mlra_w"] = base_w

            orig_node = ox.nearest_nodes(G, X=origin[1], Y=origin[0])
            dest_node = ox.nearest_nodes(G, X=dest[1],   Y=dest[0])
            path      = nx.dijkstra_path(G, orig_node, dest_node, weight="mlra_w")

            coords, blocked, total_dist = [], [], 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                ed   = G[u][v]
                k    = min(ed, key=lambda k: ed[k].get("mlra_w", 0))
                d    = ed[k]
                geom = d.get("geometry")
                if geom:
                    coords.extend((lat, lon) for lon, lat in geom.coords)
                else:
                    coords.append((G.nodes[u]["y"], G.nodes[u]["x"]))
                    coords.append((G.nodes[v]["y"], G.nodes[v]["x"]))
                total_dist += d.get("length", 0)

                ek = f"{u}_{v}_{k}"
                sg = self._segments.get(ek)
                if sg and sg.obstruction_level != ObstructionLevel.OPEN:
                    blocked.append(sg.street_name)

            speed_kmh   = {"Ambulans": 40, "İtfaiye": 35, "İş Makinesi": 20}.get(
                vehicle_type.value, 30
            )
            travel_time = (total_dist / 1000) / speed_kmh * 3600

            return {
                "coords":          coords,
                "distance_m":      total_dist,
                "travel_time_s":   travel_time,
                "blocked_streets": list(set(blocked)),
                "vehicle":         vehicle_type.value,
                "color":           VEHICLE_COLOR.get(vehicle_type.value, "#0077b6"),
            }

        except Exception as exc:
            logger.error(f"Taktik rota hatası ({vehicle_type.value}): {exc}")
            return None

    def get_visual_data(self) -> Dict:
        """Folium haritası için segmentleri kategorize eder."""
        open_segs, rest_segs, block_segs = [], [], []
        for seg in self._segments.values():
            entry = {
                "coords":        seg.coords,
                "name":          seg.street_name,
                "type":          seg.highway_type,
                "obstruction":   seg.obstruction_type,
                "texture_score": seg.texture_score,
                "passable":      [v.value for v in seg.passable_vehicles],
                "alert":         seg.alert_level,
            }
            if seg.obstruction_level == ObstructionLevel.OPEN:
                open_segs.append(entry)
            elif seg.obstruction_level == ObstructionLevel.RESTRICTED:
                rest_segs.append(entry)
            else:
                block_segs.append(entry)
        return {
            "open_roads":       open_segs,
            "restricted_roads": rest_segs,
            "blocked_roads":    block_segs,
        }

    # ── İç Metodlar ─────────────────────────────────────────

    def _sample_damage(self, coords: List[Tuple]) -> float:
        """
        Segment üzerindeki hasar piksel oranı.

        0.5m/px'de: her 50cm'lik dilim bağımsız değerlendirilebilir.
        2.5m/px'de: aynı enkaz bloğu tek piksele sığar, oran kaybolur.
        """
        samples: List[bool] = []
        for i in range(len(coords) - 1):
            p1 = self._latlon_to_px(coords[i])
            p2 = self._latlon_to_px(coords[i + 1])
            if p1 is None or p2 is None:
                continue
            n = max(4, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            for t in np.linspace(0, 1, min(n, 60)):
                cx = int(p1[0] + t * (p2[0] - p1[0]))
                cy = int(p1[1] + t * (p2[1] - p1[1]))
                for dy in range(-self._buf_px, self._buf_px + 1):
                    for dx in range(-self._buf_px, self._buf_px + 1):
                        if dx * dx + dy * dy <= self._buf_px ** 2:
                            ny, nx_ = cy + dy, cx + dx
                            if 0 <= ny < self._h and 0 <= nx_ < self._w:
                                samples.append(bool(self.binary_mask[ny, nx_]))
        return float(np.mean(samples)) if samples else 0.0

    def _analyze_texture(
        self, coords: List[Tuple]
    ) -> Tuple[float, float]:
        """
        Yol pikselleri üzerinde doku analizi.

        0.5m/px avantajı:
          Enkaz blokları (0.5–2m) → 1–16 piksel → yüksek lokal varyans.
          2.5m/px'de aynı enkaz alt-piksel detayına düşer → varyans farkı kaybolur.

        Returns:
            (texture_score ∈ [0,1], variance)
        """
        if self.gray_image is None:
            # Fallback: yalnızca hasar maskesi
            return self._sample_damage(coords), 0.0

        pixels: List[float] = []
        for i in range(len(coords) - 1):
            p1 = self._latlon_to_px(coords[i])
            p2 = self._latlon_to_px(coords[i + 1])
            if p1 is None or p2 is None:
                continue
            n = max(4, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            for t in np.linspace(0, 1, min(n, 40)):
                cx = int(p1[0] + t * (p2[0] - p1[0]))
                cy = int(p1[1] + t * (p2[1] - p1[1]))
                y0 = max(0, cy - self._buf_px)
                y1 = min(self._h, cy + self._buf_px + 1)
                x0 = max(0, cx - self._buf_px)
                x1 = min(self._w, cx + self._buf_px + 1)
                patch = self.gray_image[y0:y1, x0:x1]
                if patch.size:
                    pixels.extend(patch.flatten().tolist())

        if len(pixels) < 10:
            return 0.0, 0.0

        arr      = np.array(pixels, dtype=np.float32)
        variance = float(np.var(arr))
        # 0.5m/px kalibrasyonu: varyans eşikleri 2.5m/px'den daha dar aralıkta
        score = min(1.0, max(0.0,
            (variance - self.TEXTURE_VAR_CLEAN) /
            (self.TEXTURE_VAR_DEBRIS - self.TEXTURE_VAR_CLEAN)
        ))
        return score, variance

    def _classify(
        self,
        dmg: float, tex: float, var: float,
        dmg_thr: float, tex_thr: float,
    ) -> Tuple[ObstructionLevel, str]:
        """Hasar oranı + doku → sınıf + engel tipi."""
        combined = 0.6 * dmg + 0.4 * tex

        # Engel tipi tahmini (varyans profili)
        if dmg > 0.5 and var > 70:
            obs = "debris"
        elif var < 15 and dmg > 0.3:
            obs = "flooding"       # Homojen, düşük varyans → su
        elif dmg > 0.6 and var > 100:
            obs = "road_collapse"
        elif 0.3 < dmg <= 0.5 and 40 < var <= 80:
            obs = "vehicle_pile"
        else:
            obs = "unknown"

        if combined >= tex_thr or dmg >= 0.5:
            return ObstructionLevel.HIGH_OBSTRUCTION, obs
        if combined >= dmg_thr or dmg >= 0.2:
            return ObstructionLevel.RESTRICTED, obs
        return ObstructionLevel.OPEN, "none"

    def _passable_vehicles(
        self, level: ObstructionLevel, highway: str, width: float
    ) -> List[VehicleType]:
        """Geçiş yapabilecek araç tiplerini belirler."""
        if level == ObstructionLevel.HIGH_OBSTRUCTION:
            return []
        result = []
        for vt, c in VEHICLE_CONSTRAINTS.items():
            if highway not in c["can_use"]:
                continue
            if width < c["min_width_m"]:
                continue
            if level == ObstructionLevel.RESTRICTED and vt in (
                VehicleType.FIRE_TRUCK, VehicleType.HEAVY_MACHINERY
            ):
                continue
            result.append(vt)
        return result

    def _latlon_to_px(
        self, latlon: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        """Lat/Lon → piksel koordinatı (3 öncelikli yöntem)."""
        lat, lon = latlon

        if self.geo_transform is not None:
            try:
                from rasterio.transform import rowcol
                row, col = rowcol(self.geo_transform, lon, lat)
                if 0 <= row < self._h and 0 <= col < self._w:
                    return int(col), int(row)
            except Exception:
                pass

        if self._bbox:
            lat_min, lat_max, lon_min, lon_max = self._bbox
            if lat_max > lat_min and lon_max > lon_min:
                px = int((lon - lon_min) / (lon_max - lon_min) * self._w)
                py = int((lat_max - lat) / (lat_max - lat_min) * self._h)
                if 0 <= px < self._w and 0 <= py < self._h:
                    return px, py

        return None
