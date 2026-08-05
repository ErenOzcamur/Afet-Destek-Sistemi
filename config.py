# ============================================================
# config.py - Merkezi Konfigürasyon
# TUA Astro Hackathon DSS v2
# ============================================================

"""
Tüm eşik değerler, renkler, parametreler burada toplanır.
Hackathon süresince hızlı tweak yapabilmek için tek kaynak.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CVConfig:
    """Computer Vision pipeline parametreleri."""
    ssim_win_size: int = 7
    ssim_gaussian_weights: bool = True
    damage_threshold_low: float = 0.30
    damage_threshold_mid: float = 0.55
    damage_threshold_high: float = 0.75
    morph_kernel_size: Tuple[int, int] = (5, 5)
    min_contour_area: int = 500
    blur_kernel_size: Tuple[int, int] = (5, 5)
    max_processing_dim: int = 2048
    # Bina tespiti: min/max kontur alanı (piksel²)
    building_min_area: int = 800
    building_max_area: int = 50000
    # Yol segment buffer (piksel)
    road_buffer_px: int = 15

    def __post_init__(self) -> None:
        """Eşik değerlerin sıralı olduğunu doğrula (fail-fast)."""
        if not (self.damage_threshold_low < self.damage_threshold_mid < self.damage_threshold_high):
            raise ValueError(
                "damage_threshold degerleri sirali olmali: "
                f"low ({self.damage_threshold_low}) < mid ({self.damage_threshold_mid}) "
                f"< high ({self.damage_threshold_high})"
            )


@dataclass(frozen=True)
class MapConfig:
    """Harita görselleştirme parametreleri."""
    default_center: Tuple[float, float] = (37.75, 37.01)  # Hatay
    default_zoom: int = 13
    tile_default: str = "OpenStreetMap"           # map_builder.py tarafından kullanılır
    color_destroyed: str = "#e63946"
    color_heavy: str = "#f77f00"
    color_moderate: str = "#fcbf49"
    color_light: str = "#eae2b7"
    color_safe: str = "#06d6a0"
    color_route: str = "#0077b6"
    color_blocked_road: str = "#d62828"
    color_high: str = color_destroyed             # alias — map_builder.py tarafından kullanılır
    color_mid: str = color_heavy                  # alias — map_builder.py tarafından kullanılır
    color_low: str = color_moderate               # alias — map_builder.py tarafından kullanılır


@dataclass(frozen=True)
class RoutingConfig:
    """Rota optimizasyonu parametreleri."""
    network_type: str = "drive"
    damage_buffer_radius: float = 150.0
    weight_attribute: str = "travel_time"
    max_alternative_routes: int = 3


@dataclass(frozen=True)
class AppConfig:
    """Genel uygulama ayarları."""
    app_title: str = "🛰️ TUA Astro DSS v2"
    app_icon: str = "🛰️"
    app_layout: str = "wide"
    log_dir: str = "logs"
    export_dir: str = "exports"
    # Uzantılar küçük harf; karşılaştırmada filename.lower() kullanın (.TIF vb. için)
    supported_formats: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    log_level: str = "DEBUG"
    log_rotation: str = "10 MB"
    resolution_imece: float = 0.9          # m/pixel — İMECE uydusu
    resolution_goktürk: float = 2.5        # m/pixel — GÖKTÜRK-1 uydusu
    resolution_planet_skysat: float = 0.5  # m/pixel — Planet SkySat
    resolution_planet_ps: float = 3.0      # m/pixel — Planet PlanetScope


# Singleton instances
cv_config = CVConfig()
map_config = MapConfig()
routing_config = RoutingConfig()
app_config = AppConfig()
