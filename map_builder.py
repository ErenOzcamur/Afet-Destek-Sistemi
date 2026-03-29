# ============================================================
# map_builder.py - Harita Görselleştirme Modülü
# TUA Astro Hackathon DSS
# ============================================================

"""
Folium tabanlı interaktif harita bileşenleri.

Özellikler:
  - Before/After split-panel (kaydırıcı) karşılaştırma
  - Hasar bölgelerinin renk kodlu işaretlenmesi
  - Güvenli rota çizimi
  - Heatmap katmanı
"""

from typing import Dict, List, Optional, Tuple

import folium
import folium.plugins as folium_plugins
import numpy as np
from loguru import logger

from config import map_config


class MapBuilder:
    """
    Folium tabanlı interaktif harita oluşturucu.

    Kullanım:
        builder = MapBuilder(center=(38.35, 38.31))
        builder.add_damage_markers(regions)
        builder.add_route(route_coords)
        folium_map = builder.build()
    """

    def __init__(
        self,
        center: Tuple[float, float] = None,
        zoom: int = None,
    ):
        self.center = center or map_config.default_center
        self.zoom = zoom or map_config.default_zoom
        self._layers = []
        self._markers = []
        self._routes = []

        logger.debug(f"MapBuilder: center={self.center}, zoom={self.zoom}")

    def build(self) -> folium.Map:
        """Ana haritayı oluşturur ve tüm katmanları ekler."""
        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom,
            tiles=map_config.tile_default,
            control_scale=True,
        )

        # Uydu katmanı (toggle)
        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Esri",
            name="Uydu Görüntüsü",
            overlay=False,
        ).add_to(m)

        # Markerları ekle
        if self._markers:
            marker_group = folium.FeatureGroup(name="Hasar Bölgeleri")
            for marker_data in self._markers:
                self._create_marker(marker_data).add_to(marker_group)
            marker_group.add_to(m)

        # Rotaları ekle
        for route_data in self._routes:
            self._draw_route(route_data).add_to(m)

        # Katman kontrolü
        folium.LayerControl(collapsed=False).add_to(m)

        # MiniMap
        folium_plugins.MiniMap(
            toggle_display=True,
            tile_layer=map_config.tile_default,
        ).add_to(m)

        logger.info("Harita oluşturuldu.")
        return m

    def add_damage_markers(
        self,
        regions: List,
        transform=None,
    ) -> "MapBuilder":
        """
        Hasar bölgelerini haritaya ekler.

        Args:
            regions: DamageRegion listesi.
            transform: rasterio transform (piksel→geo dönüşüm için).

        Returns:
            self (chaining desteği)
        """
        for region in regions:
            if region.geo_centroid:
                lat, lon = region.geo_centroid
            elif transform is not None:
                try:
                    from rasterio.transform import xy
                    px_x, px_y = region.centroid
                    geo_x, geo_y = xy(transform, int(px_y), int(px_x))
                    lat, lon = geo_y, geo_x
                except Exception:
                    continue
            else:
                continue

            color = self._severity_to_color(region.severity)
            self._markers.append({
                "location": (lat, lon),
                "color": color,
                "severity": region.severity,
                "area": region.area_px,
                "score": region.severity_score,
            })

        logger.info(f"{len(self._markers)} hasar marker'ı eklendi.")
        return self

    def add_route(
        self,
        coordinates: List[Tuple[float, float]],
        color: str = None,
        label: str = "Güvenli Rota",
        weight: int = 5,
    ) -> "MapBuilder":
        """
        Rota çizgisini haritaya ekler.

        Args:
            coordinates: [(lat, lon), ...] rota noktaları.
            color: Çizgi rengi.
            label: Rota etiketi.
            weight: Çizgi kalınlığı.

        Returns:
            self (chaining desteği)
        """
        self._routes.append({
            "coordinates": coordinates,
            "color": color or map_config.color_route,
            "label": label,
            "weight": weight,
        })
        logger.debug(f"Rota eklendi: {label}")
        return self

    def _create_marker(self, data: Dict) -> folium.CircleMarker:
        """Hasar marker'ı oluşturur."""
        severity_labels = {
            "low": "Hafif Hasar",
            "mid": "Orta Hasar",
            "high": "Ağır Hasar",
        }
        popup_html = (
            f"<b>{severity_labels.get(data['severity'], 'Hasar')}</b><br>"
            f"Skor: {data['score']:.2f}<br>"
            f"Alan: {data['area']:.0f} px²"
        )

        return folium.CircleMarker(
            location=data["location"],
            radius=max(5, min(20, data["area"] / 1000)),
            color=data["color"],
            fill=True,
            fill_color=data["color"],
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=severity_labels.get(data["severity"], ""),
        )

    def _draw_route(self, data: Dict) -> folium.FeatureGroup:
        """Rota çizgisi oluşturur."""
        fg = folium.FeatureGroup(name=data["label"])

        folium.PolyLine(
            locations=data["coordinates"],
            weight=data["weight"],
            color=data["color"],
            opacity=0.8,
            tooltip=data["label"],
        ).add_to(fg)

        # Başlangıç ve bitiş marker'ları
        if data["coordinates"]:
            folium.Marker(
                location=data["coordinates"][0],
                icon=folium.Icon(color="green", icon="play"),
                tooltip="Başlangıç",
            ).add_to(fg)
            folium.Marker(
                location=data["coordinates"][-1],
                icon=folium.Icon(color="red", icon="stop"),
                tooltip="Hedef",
            ).add_to(fg)

        return fg

    def _severity_to_color(self, severity: str) -> str:
        """Hasar seviyesini renk koduna dönüştürür."""
        color_map = {
            "low": map_config.color_low,
            "mid": map_config.color_mid,
            "high": map_config.color_high,
        }
        return color_map.get(severity, map_config.color_safe)

    # --------------------------------------------------------
    # SPLIT-PANEL KARŞILAŞTIRMA
    # --------------------------------------------------------

    @staticmethod
    def create_split_map(
        center: Tuple[float, float],
        zoom: int = 15,
        left_tile: str = None,
        right_tile: str = None,
        left_label: str = "Afet Öncesi",
        right_label: str = "Afet Sonrası",
    ) -> folium.plugins.DualMap:
        """
        Before/After kaydırıcılı split-panel harita oluşturur.

        Args:
            center: Harita merkez koordinatı.
            zoom: Başlangıç zoom seviyesi.
            left_tile: Sol panel tile URL.
            right_tile: Sağ panel tile URL.
            left_label: Sol panel etiketi.
            right_label: Sağ panel etiketi.

        Returns:
            folium DualMap nesnesi.

        Not:
            Gerçek uydu görüntüleri için tile URL'lerini
            İMECE/GÖKTÜRK servis endpoint'leri ile değiştirin.
            Demo modunda OSM ve Esri Satellite kullanılır.
        """
        left_tile = left_tile or "OpenStreetMap"
        right_tile = right_tile or (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )

        dual_map = folium_plugins.DualMap(
            location=center,
            zoom_start=zoom,
            tiles=None,
        )

        # Sol panel
        folium.TileLayer(
            tiles="OpenStreetMap",
            name=left_label,
        ).add_to(dual_map.m1)

        # Sağ panel
        folium.TileLayer(
            tiles=right_tile,
            attr="Esri",
            name=right_label,
        ).add_to(dual_map.m2)

        logger.info("Split-panel harita oluşturuldu.")
        return dual_map

    @staticmethod
    def create_image_overlay_map(
        center: Tuple[float, float],
        image_before_path: str,
        image_after_path: str,
        bounds: List[List[float]],
        zoom: int = 15,
    ) -> folium.Map:
        """
        GeoTIFF/PNG görüntülerini harita üzerine overlay
        olarak yerleştirir. Opacity slider ile before/after
        karşılaştırma sağlar.

        Args:
            center: Harita merkez koordinatı.
            image_before_path: Afet öncesi görüntü dosya yolu.
            image_after_path: Afet sonrası görüntü dosya yolu.
            bounds: [[south, west], [north, east]] sınırlar.
            zoom: Başlangıç zoom seviyesi.

        Returns:
            folium.Map nesnesi.
        """
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles="OpenStreetMap",
        )

        # Before overlay
        folium.raster_layers.ImageOverlay(
            image=image_before_path,
            bounds=bounds,
            name="Afet Öncesi",
            opacity=0.7,
            interactive=True,
        ).add_to(m)

        # After overlay
        folium.raster_layers.ImageOverlay(
            image=image_after_path,
            bounds=bounds,
            name="Afet Sonrası",
            opacity=0.7,
            interactive=True,
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @staticmethod
    def create_heatmap_layer(
        coordinates_with_weights: List[Tuple[float, float, float]],
    ) -> folium_plugins.HeatMap:
        """
        Hasar yoğunluk ısı haritası katmanı.

        Args:
            coordinates_with_weights: [(lat, lon, intensity), ...]

        Returns:
            folium HeatMap plugin.
        """
        return folium_plugins.HeatMap(
            data=coordinates_with_weights,
            name="Hasar Yoğunluk Haritası",
            min_opacity=0.4,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={
                0.2: "#ffffb2",
                0.4: "#fecc5c",
                0.6: "#fd8d3c",
                0.8: "#f03b20",
                1.0: "#bd0026",
            },
        )
