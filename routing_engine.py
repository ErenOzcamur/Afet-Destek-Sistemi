# ============================================================
# routing_engine.py - Lojistik Rota Optimizasyonu
# TUA Astro Hackathon DSS
# ============================================================

"""
Hasar tespiti sonuçlarını lojistik planlamaya entegre eden
rota optimizasyon modülü.

Yaklaşım:
─────────
1. Hasarlı bölge koordinatlarını OSMnx yol ağı grafiğine
   'engel' (obstacle) olarak işle.
2. Engelli kenarların (edge) ağırlıklarını artırarak (veya
   tamamen kaldırarak) güvenli rota hesapla.
3. Dijkstra / A* ile en kısa güvenli yolu bul.
4. Alternatif rotalar oluştur (k-shortest paths).

Not:
────
OSMnx, OpenStreetMap verisi üzerinden çalışır.
Gerçek zamanlı trafik verisi içermez.
48 saatlik sprint için yeterli bir yaklaşımdır.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from config import routing_config


class SafeRouteOptimizer:
    """
    Hasarlı bölgeleri engel olarak işaretleyip güvenli
    yardım ulaşım rotası hesaplayan sınıf.

    Kullanım:
        optimizer = SafeRouteOptimizer()
        graph = optimizer.load_network(center=(38.35, 38.31), dist=5000)
        optimizer.mark_damaged_zones(damage_coordinates)
        route = optimizer.find_safe_route(origin, destination)
    """

    def __init__(self, config=None):
        self.config = config or routing_config
        self.graph = None
        self.damaged_edges = set()
        self.blocked_edges = set()
        self._osmnx = None
        self._nx = None

        # Lazy import
        self._import_dependencies()

        logger.info("SafeRouteOptimizer başlatıldı.")

    def _import_dependencies(self):
        """Bağımlılıkları lazy-import et."""
        try:
            import osmnx as ox
            import networkx as nx
            self._osmnx = ox
            self._nx = nx
            logger.debug("osmnx ve networkx yüklendi.")
        except ImportError as e:
            logger.warning(
                f"Rota optimizasyonu bağımlılıkları eksik: {e}. "
                f"Routing fonksiyonları devre dışı."
            )

    def load_network(
        self,
        center: Tuple[float, float],
        dist: int = 5000,
        network_type: str = None,
    ) -> Optional[object]:
        """
        Belirtilen merkez ve yarıçap için OSM yol ağını indirir.

        Args:
            center: (latitude, longitude) merkez noktası.
            dist: Yarıçap (metre).
            network_type: "drive", "walk", "bike", "all".

        Returns:
            NetworkX MultiDiGraph veya None.
        """
        if self._osmnx is None:
            logger.error("osmnx yüklü değil.")
            return None

        network_type = network_type or self.config.network_type

        try:
            logger.info(
                f"Yol ağı indiriliyor: center={center}, "
                f"dist={dist}m, type={network_type}"
            )
            self.graph = self._osmnx.graph_from_point(
                center,
                dist=dist,
                network_type=network_type,
            )

            # Seyahat süresi hesapla (hız bilgisine göre)
            self.graph = self._osmnx.add_edge_speeds(self.graph, fallback=50)
            self.graph = self._osmnx.add_edge_travel_times(self.graph)

            node_count = self.graph.number_of_nodes()
            edge_count = self.graph.number_of_edges()
            logger.info(
                f"Yol ağı yüklendi: {node_count} düğüm, "
                f"{edge_count} kenar."
            )
            return self.graph

        except Exception as e:
            logger.error(f"Yol ağı yükleme hatası: {e}")
            return None

    def mark_damaged_zones(
        self,
        damage_coords: List[Tuple[float, float]],
        buffer_radius: float = None,
    ) -> int:
        """
        Hasarlı bölge koordinatlarını engel olarak işaretle.

        Buffer radius içindeki tüm yol segmentlerinin (edge)
        ağırlığını büyük bir penaltı ile artırır.

        Args:
            damage_coords: [(lat, lon), ...] hasar noktaları.
            buffer_radius: Etki yarıçapı (metre).

        Returns:
            Etkilenen kenar sayısı.
        """
        if self.graph is None:
            logger.error("Önce load_network() çağrılmalı.")
            return 0

        buffer_radius = buffer_radius or self.config.damage_buffer_radius
        affected_count = 0

        for lat, lon in damage_coords:
            try:
                # En yakın düğümü bul
                nearest_node = self._osmnx.nearest_nodes(
                    self.graph, lon, lat
                )

                # Buffer içindeki tüm düğümleri bul
                ego_graph = self._nx.ego_graph(
                    self.graph,
                    nearest_node,
                    radius=buffer_radius,
                    distance="length",
                )

                # Etkilenen kenarları işaretle
                for u, v, key in ego_graph.edges(keys=True):
                    if self.graph.has_edge(u, v, key):
                        # Penaltı: seyahat süresini 100x artır
                        original_tt = self.graph[u][v][key].get(
                            "travel_time", 60
                        )
                        self.graph[u][v][key]["travel_time"] = (
                            original_tt * 100
                        )
                        self.graph[u][v][key]["damaged"] = True
                        self.damaged_edges.add((u, v, key))
                        affected_count += 1

            except Exception as e:
                logger.warning(
                    f"Hasar işaretleme hatası ({lat}, {lon}): {e}"
                )

        logger.info(
            f"{len(damage_coords)} hasar noktası işlendi. "
            f"{affected_count} kenar etkilendi."
        )
        return affected_count

    def find_safe_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        weight: str = None,
    ) -> Optional[Dict]:
        """
        Hasarlı bölgeleri atlayarak güvenli rota hesaplar.

        Args:
            origin: (lat, lon) başlangıç noktası.
            destination: (lat, lon) bitiş noktası.
            weight: Ağırlık niteliği ("travel_time" veya "length").

        Returns:
            Dict:
                - 'nodes': Rota düğüm listesi
                - 'coordinates': [(lat, lon), ...] rota koordinatları
                - 'total_length_m': Toplam uzunluk (metre)
                - 'total_time_s': Toplam süre (saniye)
                - 'passes_damage': Hasarlı bölgeden geçiyor mu?
            Veya None (rota bulunamazsa).
        """
        if self.graph is None:
            logger.error("Önce load_network() çağrılmalı.")
            return None

        weight = weight or self.config.weight_attribute

        try:
            # En yakın düğümleri bul
            orig_node = self._osmnx.nearest_nodes(
                self.graph, origin[1], origin[0]
            )
            dest_node = self._osmnx.nearest_nodes(
                self.graph, destination[1], destination[0]
            )

            # En kısa yol (Dijkstra)
            route_nodes = self._nx.shortest_path(
                self.graph,
                orig_node,
                dest_node,
                weight=weight,
            )

            # Rota koordinatlarını çıkar
            coordinates = []
            for node in route_nodes:
                node_data = self.graph.nodes[node]
                coordinates.append(
                    (node_data["y"], node_data["x"])  # (lat, lon)
                )

            # Toplam uzunluk ve süre
            total_length = 0.0
            total_time = 0.0
            passes_damage = False

            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i + 1]
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    total_length += edge_data[first_key].get("length", 0)
                    total_time += edge_data[first_key].get(
                        "travel_time", 0
                    )
                    if edge_data[first_key].get("damaged", False):
                        passes_damage = True

            result = {
                "nodes": route_nodes,
                "coordinates": coordinates,
                "total_length_m": round(total_length, 1),
                "total_time_s": round(total_time, 1),
                "passes_damage": passes_damage,
            }

            logger.info(
                f"Rota hesaplandı: {total_length:.0f}m, "
                f"{total_time:.0f}s, "
                f"Hasarlı: {'EVET' if passes_damage else 'HAYIR'}"
            )
            return result

        except self._nx.NetworkXNoPath:
            logger.warning(
                "Hedefe ulaşılabilir güvenli rota bulunamadı."
            )
            return None
        except Exception as e:
            logger.error(f"Rota hesaplama hatası: {e}")
            return None

    def find_alternative_routes(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        k: int = None,
    ) -> List[Dict]:
        """
        k adet alternatif rota hesaplar (Yen's k-shortest paths).

        Args:
            origin: Başlangıç koordinatı.
            destination: Bitiş koordinatı.
            k: Alternatif rota sayısı.

        Returns:
            Rota dict'lerinin listesi (en kısadan uzuna).
        """
        if self.graph is None or self._nx is None:
            logger.error("Graph yüklü değil.")
            return []

        k = k or self.config.max_alternative_routes
        weight = self.config.weight_attribute

        try:
            orig_node = self._osmnx.nearest_nodes(
                self.graph, origin[1], origin[0]
            )
            dest_node = self._osmnx.nearest_nodes(
                self.graph, destination[1], destination[0]
            )

            # k-shortest paths
            paths = list(
                self._nx.shortest_simple_paths(
                    self.graph, orig_node, dest_node, weight=weight,
                )
            )

            routes = []
            for i, path in enumerate(paths[:k]):
                coordinates = []
                total_length = 0.0

                for node in path:
                    nd = self.graph.nodes[node]
                    coordinates.append((nd["y"], nd["x"]))

                for j in range(len(path) - 1):
                    edge = self.graph.get_edge_data(path[j], path[j + 1])
                    if edge:
                        fk = list(edge.keys())[0]
                        total_length += edge[fk].get("length", 0)

                routes.append({
                    "route_id": i + 1,
                    "nodes": path,
                    "coordinates": coordinates,
                    "total_length_m": round(total_length, 1),
                })

            logger.info(f"{len(routes)} alternatif rota hesaplandı.")
            return routes

        except Exception as e:
            logger.error(f"Alternatif rota hatası: {e}")
            return []

    # ════════════════════════════════════════════════════════
    # GÖRÜNTÜ-YOL HASAR ANALİZİ
    # ════════════════════════════════════════════════════════

    def analyze_road_damage(
        self,
        binary_mask: np.ndarray,
        geo_transform=None,
        damage_threshold: float = 0.30,
    ) -> Dict[str, dict]:
        """
        Binary hasar maskesini OSM yol ağıyla piksel bazında çakıştırır.

        Yerli Uydu Avantajı (İMECE/GÖKTÜRK):
        ─────────────────────────────────────
        İMECE'nin sub-metre (<1m) GSD'si ile bir yol şeridi üzerindeki
        2-3 m genişliğindeki enkaz, çatlak veya su taşkını tespit
        edilebilir. Sentinel-2'nin 10m çözünürlüğünde aynı engeller
        mixed-pixel problemine girerek gözden kaçar. Sonuç:
          - Tek şerit kapanması (3m enkaz) → İMECE: ✅ tespit
          - Aynı senaryo → Sentinel-2: ❌ gözden kaçar
          - Kavşak hasarı → İMECE: ayrıntılı kontur, Sentinel: belirsiz

        Yöntem:
        1. OSM kenarlarının geometrisini alır
        2. Geo → piksel dönüşümü (transform veya bbox)
        3. Bresenham örnekleme: her yol segmenti boyunca piksel okur
        4. Hasar piksel oranı = damaged_pts / total_pts

        Args:
            binary_mask: OpenCV hasar maskesi (HxW, uint8, 0/255)
            geo_transform: rasterio Affine transform (opsiyonel)
            damage_threshold: Engelleme eşiği (varsayılan %30)

        Returns:
            {osm_id: {road_name, damage_ratio, status, length_m, geometry_coords, edge_key}}
        """
        if self.graph is None:
            logger.error("analyze_road_damage: Önce load_network() çağrılmalı.")
            return {}

        try:
            edges = self._osmnx.graph_to_gdfs(self.graph, nodes=False)
        except Exception as e:
            logger.error(f"Edge verisi alınamadı: {e}")
            return {}

        h, w = binary_mask.shape[:2]

        # Bounding box hesapla (geo→pixel doğrusal interpolasyon için)
        bbox = None
        try:
            nodes_gdf = self._osmnx.graph_to_gdfs(self.graph, edges=False)
            bbox = (
                float(nodes_gdf.geometry.x.min()),
                float(nodes_gdf.geometry.y.min()),
                float(nodes_gdf.geometry.x.max()),
                float(nodes_gdf.geometry.y.max()),
            )
            logger.debug(f"Yol ağı bbox: {bbox}")
        except Exception:
            pass

        road_damage: Dict[str, dict] = {}

        for (u, v, k), edge in edges.iterrows():
            osm_id = str(edge.get("osmid", f"{u}_{v}_{k}"))
            name = edge.get("name", "İsimsiz Yol")
            if isinstance(name, list):
                name = name[0] if name else "İsimsiz Yol"
            length_m = float(edge.get("length", 0))

            geom = edge.get("geometry")
            if geom is None:
                try:
                    n1 = self.graph.nodes[u]
                    n2 = self.graph.nodes[v]
                    from shapely.geometry import LineString
                    geom = LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])])
                except Exception:
                    continue

            coords = list(geom.coords)  # [(lon, lat), ...]
            damage_ratio = self._sample_road_damage(
                coords, binary_mask, h, w, geo_transform, bbox
            )

            if damage_ratio >= damage_threshold:
                status = "BLOCKED"
            elif damage_ratio >= 0.10:
                status = "DAMAGED"
            else:
                status = "OPEN"

            road_damage[osm_id] = {
                "road_name": name,
                "damage_ratio": round(damage_ratio, 3),
                "status": status,
                "length_m": length_m,
                "geometry_coords": [(c[1], c[0]) for c in coords],  # (lat, lon)
                "edge_key": (u, v, k),
            }

        blocked = sum(1 for d in road_damage.values() if d["status"] == "BLOCKED")
        damaged = sum(1 for d in road_damage.values() if d["status"] == "DAMAGED")
        open_ = len(road_damage) - blocked - damaged
        logger.info(
            f"Yol hasar analizi: toplam={len(road_damage)} | "
            f"BLOCKED={blocked} DAMAGED={damaged} OPEN={open_}"
        )

        self._road_damage_cache = road_damage
        return road_damage

    def _sample_road_damage(
        self,
        coords: List[Tuple[float, float]],
        mask: np.ndarray,
        h: int,
        w: int,
        geo_transform=None,
        bbox=None,
    ) -> float:
        """Yol geometrisi boyunca piksel örnekleyerek hasar oranı hesaplar."""
        total_pts = 0
        damaged_pts = 0

        for i in range(len(coords) - 1):
            lon1, lat1 = float(coords[i][0]), float(coords[i][1])
            lon2, lat2 = float(coords[i + 1][0]), float(coords[i + 1][1])

            px1 = self._geo_to_pixel(lat1, lon1, (h, w), geo_transform, bbox)
            px2 = self._geo_to_pixel(lat2, lon2, (h, w), geo_transform, bbox)

            steps = max(abs(px2[0] - px1[0]), abs(px2[1] - px1[1]), 1)
            steps = min(steps, 80)  # Maksimum 80 örnek / segment

            for t in range(steps + 1):
                px = int(px1[0] + (px2[0] - px1[0]) * t / steps)
                py = int(px1[1] + (px2[1] - px1[1]) * t / steps)
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                total_pts += 1
                if mask[py, px] > 0:
                    damaged_pts += 1

        return damaged_pts / max(total_pts, 1)

    def _geo_to_pixel(
        self,
        lat: float,
        lon: float,
        mask_shape: Tuple[int, int],
        geo_transform=None,
        bbox=None,
    ) -> Tuple[int, int]:
        """
        Coğrafi koordinatı piksel koordinatına dönüştürür.

        Öncelik sırası:
          1. rasterio Affine transform (GeoTIFF meta, en doğru)
          2. OSM bbox lineer interpolasyon (demo mod)
          3. Global normalizasyon (fallback)
        """
        h, w = mask_shape[:2]

        if geo_transform is not None:
            try:
                from rasterio.transform import rowcol
                row, col = rowcol(geo_transform, lon, lat)
                return (int(np.clip(col, 0, w - 1)), int(np.clip(row, 0, h - 1)))
            except Exception:
                pass

        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            if max_lon > min_lon and max_lat > min_lat:
                px_x = (lon - min_lon) / (max_lon - min_lon) * w
                px_y = (max_lat - lat) / (max_lat - min_lat) * h
                return (int(np.clip(px_x, 0, w - 1)), int(np.clip(px_y, 0, h - 1)))

        # Fallback: kaba global normalizasyon
        px_x = int(((lon + 180) % 360) / 360 * w) % w
        px_y = int(((90 - lat) % 180) / 180 * h) % h
        return (px_x % w, px_y % h)

    def apply_damage_weights(
        self,
        road_damage: Dict,
        block_threshold: float = 0.30,
        penalty_threshold: float = 0.10,
    ) -> int:
        """
        Hasar oranına göre yol ağırlıklarını günceller ve BLOCKED yolları kapatır.

        Ağırlıklandırma Stratejisi (Yerli Uydu Verisi Entegrasyonu):
        ─────────────────────────────────────────────────────────────
        İMECE'nin yüksek çözünürlüğü yol yüzeyini ve genişliğini
        analiz etmeye imkân tanır. Bu bilgi algoritmaya penaltı
        puanı olarak eklenir:

          BLOCKED  (hasar > %30) → weight = 9.999.999 (efektif ∞)
            Dijkstra bu segmenti kesinlikle atlar.
          DAMAGED  (%10-%30)     → weight × 2 (yavaş geçiş)
            Acil yardım araçları dikkatli geçebilir.
          NARROW   (dar sokak)   → weight × 3
            İş makinesi, büyük kamyon geçemez.
            (İMECE ile yol genişliği ~1m hassasiyetle ölçülebilir)
          OPEN     (hasar < %10) → değişmez

        Args:
            road_damage: analyze_road_damage() çıktısı
            block_threshold: Engelleme eşiği (varsayılan %30)
            penalty_threshold: Penaltı eşiği (varsayılan %10)

        Returns:
            BLOCKED yol segmenti sayısı
        """
        if self.graph is None:
            return 0

        BLOCKED_WEIGHT = 9_999_999
        NARROW_PENALTY = 3.0
        NARROW_TYPES = {"residential", "footway", "path", "track", "service", "unclassified"}

        blocked_count = 0

        for osm_id, data in road_damage.items():
            u, v, k = data.get("edge_key", (None, None, None))
            if u is None or not self.graph.has_edge(u, v, k):
                continue

            damage_ratio = data["damage_ratio"]
            ed = self.graph[u][v][k]

            if damage_ratio >= block_threshold:
                # BLOCKED: algoritmik ∞ ağırlık
                ed["travel_time"] = BLOCKED_WEIGHT
                ed["length"] = BLOCKED_WEIGHT
                ed["road_status"] = "BLOCKED"
                self.blocked_edges.add((u, v, k))
                blocked_count += 1

            elif damage_ratio >= penalty_threshold:
                # DAMAGED: 2× penaltı
                for attr in ("travel_time", "length"):
                    orig = ed.get(attr, 60)
                    if orig < BLOCKED_WEIGHT:
                        ed[attr] = orig * 2
                ed["road_status"] = "DAMAGED"

            else:
                ed["road_status"] = "OPEN"

            # Dar yol penaltısı (İMECE ile tespit edilebilir)
            highway = ed.get("highway", "")
            if isinstance(highway, list):
                highway = highway[0] if highway else ""
            if highway in NARROW_TYPES:
                for attr in ("travel_time", "length"):
                    curr = ed.get(attr, 60)
                    if curr < BLOCKED_WEIGHT:
                        ed[attr] = curr * NARROW_PENALTY

        logger.info(
            f"Ağırlıklar güncellendi: {blocked_count} yol BLOCKED, "
            f"{len(road_damage) - blocked_count} yol açık/hasarlı."
        )
        return blocked_count

    def find_alternative_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> Optional[Dict]:
        """
        İkinci güvenli rotayı hesaplar (birincil rotanın alternatifi).

        Sunumda kullanım:
        "Sistemimiz kapalı yolları atlayarak otomatik olarak
         2. dakika daha uzun ama güvenli olan alternatif rotayı
         AFAD ekiplerine iletti."

        Returns:
            find_safe_route() ile aynı formatta Dict veya None
        """
        routes = self.find_alternative_routes(origin, destination, k=3)
        # En az 2 farklı rota varsa 2.'yi al
        if len(routes) >= 2:
            alt = routes[1]
            logger.info(
                f"Alternatif rota: {alt['total_length_m']:.0f}m "
                f"(+{alt['total_length_m'] - routes[0]['total_length_m']:.0f}m birincil rotaya göre)"
            )
            # find_alternative_routes formatını find_safe_route formatına çevir
            return {
                "coordinates": alt.get("coordinates", []),
                "total_length_m": alt.get("total_length_m", 0),
                "total_time_s": alt.get("total_length_m", 0) / 8.33,  # ~30 km/h
                "passes_damage": False,  # Yen's k-shortest hasarlı yollardan kaçar
            }
        elif routes:
            return {
                "coordinates": routes[0].get("coordinates", []),
                "total_length_m": routes[0].get("total_length_m", 0),
                "total_time_s": routes[0].get("total_length_m", 0) / 8.33,
                "passes_damage": False,
            }
        return None

    def get_road_visual_data(self) -> Dict:
        """
        Harita görselleştirmesi için yol verilerini kategorize eder.

        Returns:
            {
                'open_roads':    [{'coords': [(lat,lon),...], 'name': str}],
                'blocked_roads': [{'coords': [(lat,lon),...], 'name': str, 'damage_ratio': float}],
                'damaged_roads': [{'coords': [(lat,lon),...], 'name': str, 'damage_ratio': float}],
            }
        """
        if self.graph is None or not hasattr(self, "_road_damage_cache"):
            logger.warning("get_road_visual_data: veri yok, önce analyze_road_damage() çağrılmalı.")
            return {"open_roads": [], "blocked_roads": [], "damaged_roads": []}

        result: Dict[str, list] = {
            "open_roads": [],
            "blocked_roads": [],
            "damaged_roads": [],
        }

        for osm_id, data in self._road_damage_cache.items():
            entry = {
                "coords": data["geometry_coords"],
                "name": data["road_name"],
                "damage_ratio": data["damage_ratio"],
            }
            status = data.get("status", "OPEN")
            if status == "BLOCKED":
                result["blocked_roads"].append(entry)
            elif status == "DAMAGED":
                result["damaged_roads"].append(entry)
            else:
                result["open_roads"].append(entry)

        logger.debug(
            f"Görsel veri: {len(result['open_roads'])} açık | "
            f"{len(result['damaged_roads'])} hasarlı | "
            f"{len(result['blocked_roads'])} kapalı"
        )
        return result

    def get_network_stats(self) -> Optional[Dict]:
        """Yol ağı istatistikleri."""
        if self.graph is None:
            return None

        stats = self._osmnx.basic_stats(self.graph)
        stats["damaged_edges"] = len(self.damaged_edges)
        stats["total_edges"] = self.graph.number_of_edges()
        stats["damage_ratio"] = (
            len(self.damaged_edges) / self.graph.number_of_edges()
            if self.graph.number_of_edges() > 0 else 0
        )
        return stats
