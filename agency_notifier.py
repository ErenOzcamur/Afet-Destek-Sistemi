# ============================================================
# agency_notifier.py - Kurumsal Olay & Bildirim Motoru
# TUA Astro Hackathon DSS — AFAD / UMKE / Belediye / İtfaiye
# ============================================================

"""
Micro-Logistic Road Assessment (MLRA) çıktısını kurumsal
bildirimlere dönüştürür.

AlertPayload şeması (kullanıcı tarafından belirlenmiş):
    alert_level       : CRITICAL | WARNING | INFO
    obstruction_type  : Debris | Flooding | Road Collapse | Vehicle Pile | Unknown
    location          : {lat, lon, street_name}
    recommended_action: string
"""

from __future__ import annotations

import hashlib
import html
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Tuple

from loguru import logger

# ── Rapor Renk Sabitleri ────────────────────────────────────

COLOR_CRITICAL = "#e63946"
COLOR_WARNING = "#f77f00"
BG_CRITICAL = "rgba(230,57,70,0.08)"
BG_WARNING = "rgba(247,127,0,0.08)"


def _dump_json(data: Dict, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)


# ── Enum'lar ────────────────────────────────────────────────

class AlertLevel(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"


class ObstructionType(str, Enum):
    DEBRIS        = "Debris"
    FLOODING      = "Flooding"
    ROAD_COLLAPSE = "Road Collapse"
    VEHICLE_PILE  = "Vehicle Pile"
    UNKNOWN       = "Unknown"


class TargetAgency(str, Enum):
    AFAD     = "AFAD"
    UMKE     = "UMKE"
    BELEDIYE = "Belediye"
    ITFAIYE  = "İtfaiye"


# ── Haritalama Tabloları ────────────────────────────────────

OBSTRUCTION_MAP: Dict[str, ObstructionType] = {
    "debris":        ObstructionType.DEBRIS,
    "flooding":      ObstructionType.FLOODING,
    "road_collapse": ObstructionType.ROAD_COLLAPSE,
    "vehicle_pile":  ObstructionType.VEHICLE_PILE,
    "unknown":       ObstructionType.UNKNOWN,
    "none":          ObstructionType.UNKNOWN,
}

# Engel tipine göre hedef kurum
AGENCY_ROUTING: Dict[ObstructionType, List[TargetAgency]] = {
    ObstructionType.DEBRIS:        [TargetAgency.AFAD, TargetAgency.BELEDIYE],
    ObstructionType.FLOODING:      [TargetAgency.AFAD, TargetAgency.BELEDIYE],
    ObstructionType.ROAD_COLLAPSE: [TargetAgency.AFAD, TargetAgency.UMKE],
    ObstructionType.VEHICLE_PILE:  [TargetAgency.BELEDIYE, TargetAgency.ITFAIYE],
    ObstructionType.UNKNOWN:       [TargetAgency.AFAD],
}

# (alert_level, obstruction_type) → önerilen eylem
RECOMMENDED_ACTIONS: Dict[Tuple[str, str], str] = {
    ("CRITICAL", "Debris"):        "Ağır iş makinesi gönder — enkaz kaldırma ekibi talep et",
    ("CRITICAL", "Flooding"):      "Su tahliye pompası + bot görevlendir — geçişi kapat",
    ("CRITICAL", "Road Collapse"): "Geoteknik ekip çağır — şerit tamamen kapat",
    ("CRITICAL", "Vehicle Pile"):  "Çekici + trafik zabıtası gönder — koridoru temizle",
    ("WARNING",  "Debris"):        "Hafif temizlik ekibi gönder — binek araç geçişine izin ver",
    ("WARNING",  "Flooding"):      "Yol kapanmadan tahliye et — alternatif rota belirt",
    ("WARNING",  "Road Collapse"): "Yol bakım ekibi gönder — ağır araç geçişini kısıtla",
    ("WARNING",  "Vehicle Pile"):  "Trafik akışını yönlendir — tıkanıklık gider",
}


# ── Dataclass'lar ───────────────────────────────────────────

@dataclass
class IncidentLocation:
    lat: float
    lon: float
    street_name: str
    highway_type: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "lat":          round(self.lat, 6),
            "lon":          round(self.lon, 6),
            "street_name":  self.street_name,
            "highway_type": self.highway_type,
            "maps_link": (
                f"https://www.openstreetmap.org/"
                f"?mlat={self.lat}&mlon={self.lon}&zoom=18"
            ),
        }


@dataclass
class AlertPayload:
    """
    Tek bir yol kapanması olayı için kurumsal bildirim paketi.

    Kullanıcı tarafından belirlenen şema:
        alert_level       : CRITICAL | WARNING | INFO
        obstruction_type  : Debris | Flooding | Road Collapse | ...
        location          : {lat, lon, street_name}
        recommended_action: string
    """
    incident_id:        str
    alert_level:        AlertLevel
    obstruction_type:   ObstructionType
    location:           IncidentLocation
    recommended_action: str
    target_agencies:    List[TargetAgency]
    texture_score:      float
    pixel_damage_ratio: float
    passable_vehicles:  List[str]
    timestamp_utc:      str
    satellite_source:   str = "Planet-SkySat"

    def to_dict(self) -> Dict:
        return {
            "incident_id":       self.incident_id,
            "timestamp_utc":     self.timestamp_utc,
            "satellite_source":  self.satellite_source,
            "alert_level":       self.alert_level.value,
            "obstruction_type":  self.obstruction_type.value,
            "location":          self.location.to_dict(),
            "recommended_action": self.recommended_action,
            "target_agencies":   [a.value for a in self.target_agencies],
            "passable_vehicles": self.passable_vehicles,
            "technical": {
                "texture_score":      round(self.texture_score, 3),
                "pixel_damage_ratio": round(self.pixel_damage_ratio, 3),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return _dump_json(self.to_dict(), indent=indent)


@dataclass
class IncidentReport:
    """Tüm olayları kapsayan kurumsal rapor nesnesi."""
    report_id:       str
    generated_at:    str
    satellite_source: str
    center_coords:   Tuple[float, float]
    total_incidents: int
    critical_count:  int
    warning_count:   int
    payloads:        List[AlertPayload] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "report_id":       self.report_id,
            "generated_at":    self.generated_at,
            "satellite_source": self.satellite_source,
            "center_coords":   {
                "lat": self.center_coords[0],
                "lon": self.center_coords[1],
            },
            "total_incidents": self.total_incidents,
            "critical_count":  self.critical_count,
            "warning_count":   self.warning_count,
        }

    def to_json(self, indent: int = 2) -> str:
        data = self.summary()
        data["incidents"] = [p.to_dict() for p in self.payloads]
        return _dump_json(data, indent=indent)


# ── Ana Sınıf ───────────────────────────────────────────────

class AgencyNotifier:
    """
    RoadSegment sözlüğünü → IncidentReport + AlertPayload listesine
    dönüştürür; HTML rapor ve JSON export üretir.
    """

    def __init__(self, satellite_source: str = "Planet-SkySat"):
        self.satellite_source = satellite_source

    def build_report(
        self,
        segments: Dict,                       # edge_key → RoadSegment
        center_coords: Tuple[float, float],
        min_alert_level: AlertLevel = AlertLevel.WARNING,
    ) -> IncidentReport:
        """
        MLRAEngine.analyze_all_segments() çıktısından kurumsal rapor oluşturur.

        Args:
            segments:         MLRA segment sözlüğü
            center_coords:    Harita merkezi (lat, lon)
            min_alert_level:  Minimum bildirim seviyesi
        """
        from planet_engine import ObstructionLevel   # lazy import — döngüsel bağımlılık yok

        payloads: List[AlertPayload] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for seg in segments.values():
            if seg.obstruction_level == ObstructionLevel.OPEN:
                continue

            try:
                level = AlertLevel(seg.alert_level)
            except ValueError:
                logger.warning(
                    f"AgencyNotifier: geçersiz alert_level atlandı: "
                    f"{seg.alert_level!r} ({getattr(seg, 'street_name', '?')})"
                )
                continue

            # Minimum seviye filtresi
            if not self._passes_min_level(level, min_alert_level):
                continue

            payloads.append(self._build_payload(seg, level, now_utc))

        # CRITICAL önce (yeni liste — mutasyon yok)
        payloads = sorted(
            payloads, key=lambda p: 0 if p.alert_level == AlertLevel.CRITICAL else 1
        )

        n_crit = sum(1 for p in payloads if p.alert_level == AlertLevel.CRITICAL)
        n_warn = sum(1 for p in payloads if p.alert_level == AlertLevel.WARNING)
        rep_id = f"MLRA-{datetime.now(timezone.utc):%Y%m%d%H%M}-{uuid.uuid4().hex[:6].upper()}"

        logger.info(
            f"AgencyNotifier: {len(payloads)} olay — "
            f"{n_crit} kritik, {n_warn} uyarı"
        )

        return IncidentReport(
            report_id        = rep_id,
            generated_at     = now_utc,
            satellite_source = self.satellite_source,
            center_coords    = center_coords,
            total_incidents  = len(payloads),
            critical_count   = n_crit,
            warning_count    = n_warn,
            payloads         = payloads,
        )

    def to_html_report(self, report: IncidentReport) -> str:
        """Print-to-PDF hazır, kurumsal HTML rapor üretir."""
        now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

        row_parts: List[str] = []
        for i, p in enumerate(report.payloads, 1):
            lc  = COLOR_CRITICAL if p.alert_level == AlertLevel.CRITICAL else COLOR_WARNING
            lbg = BG_CRITICAL if p.alert_level == AlertLevel.CRITICAL else BG_WARNING
            ag  = html.escape(", ".join(a.value for a in p.target_agencies))
            pv  = html.escape(", ".join(p.passable_vehicles) or "—")
            row_parts.append(f"""
            <tr style="background:{lbg}">
              <td style="text-align:center;font-weight:600">{i}</td>
              <td><span style="background:{lc};color:#fff;padding:2px 8px;
                  border-radius:4px;font-weight:700;font-size:.82em">
                  {p.alert_level.value}</span></td>
              <td>{html.escape(p.location.street_name)}</td>
              <td>{html.escape(p.obstruction_type.value)}</td>
              <td style="font-size:.83em">{html.escape(p.recommended_action)}</td>
              <td style="font-size:.8em">{ag}</td>
              <td style="font-size:.8em">{pv}</td>
            </tr>""")
        rows = "".join(row_parts)

        src_short = report.satellite_source.split("-")[-1] \
                    if "-" in report.satellite_source else report.satellite_source

        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>MLRA Raporu — {report.report_id}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',system-ui,sans-serif;background:#f8f9fa;color:#1a1a2e;padding:32px}}
  .hdr{{background:linear-gradient(135deg,#0f0c29,#302b63);color:#fff;
        padding:24px 32px;border-radius:12px;margin-bottom:20px}}
  .hdr h1{{font-size:1.55em;font-weight:800;margin-bottom:4px}}
  .hdr p{{opacity:.72;font-size:.88em;margin-top:4px}}
  .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
  .kpi{{background:#fff;border-radius:10px;padding:16px;text-align:center;
        box-shadow:0 2px 8px rgba(0,0,0,.06)}}
  .kpi .v{{font-size:2em;font-weight:800}}
  .kpi .l{{font-size:.78em;color:#666;margin-top:4px}}
  .red .v{{color:{COLOR_CRITICAL}}} .orange .v{{color:{COLOR_WARNING}}} .green .v{{color:#06d6a0}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
         overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
  th{{background:#1a1a2e;color:#fff;padding:10px 14px;font-size:.83em;text-align:left}}
  td{{padding:9px 14px;border-bottom:1px solid #f0f0f0;font-size:.83em;vertical-align:middle}}
  tr:last-child td{{border-bottom:none}}
  .ftr{{margin-top:20px;text-align:center;font-size:.75em;color:#aaa}}
  @media print{{
    body{{background:#fff;padding:16px}}
    .hdr,.kpi th{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  }}
</style>
</head>
<body>
<div class="hdr">
  <h1>🛰️ MLRA — Mikro-Lojistik Yol Değerlendirme Raporu</h1>
  <p>Rapor: <b>{report.report_id}</b> &nbsp;|&nbsp; {now_str}
     &nbsp;|&nbsp; Kaynak: {report.satellite_source}</p>
  <p>Merkez: {report.center_coords[0]:.4f}°K, {report.center_coords[1]:.4f}°D</p>
</div>
<div class="kpis">
  <div class="kpi red"><div class="v">{report.critical_count}</div>
       <div class="l">KRİTİK OLAY</div></div>
  <div class="kpi orange"><div class="v">{report.warning_count}</div>
       <div class="l">UYARI</div></div>
  <div class="kpi"><div class="v">{report.total_incidents}</div>
       <div class="l">TOPLAM OLAY</div></div>
  <div class="kpi green"><div class="v">{src_short}</div>
       <div class="l">UYDU KAYNAĞI</div></div>
</div>
<table>
  <thead>
    <tr>
      <th>#</th><th>Seviye</th><th>Sokak</th><th>Engel Tipi</th>
      <th>Önerilen Eylem</th><th>Hedef Kurum</th><th>Geçiş İzni</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
<div class="ftr">TUA Astro Hackathon 2026 — ASTRO-RESQ DSS v2 — {report.satellite_source}</div>
</body>
</html>"""

    # ── Yardımcı ────────────────────────────────────────────

    @staticmethod
    def _passes_min_level(level: AlertLevel, min_alert_level: AlertLevel) -> bool:
        """Minimum bildirim seviyesi filtresi."""
        if min_alert_level == AlertLevel.CRITICAL and level != AlertLevel.CRITICAL:
            return False
        if min_alert_level == AlertLevel.WARNING and level == AlertLevel.INFO:
            return False
        return True

    def _build_payload(self, seg, level: AlertLevel, now_utc: str) -> AlertPayload:
        """Tek segmentten AlertPayload üretir."""
        if seg.obstruction_type not in OBSTRUCTION_MAP:
            logger.warning(
                f"AgencyNotifier: bilinmeyen obstruction_type "
                f"{seg.obstruction_type!r} — UNKNOWN olarak işaretlendi"
            )
        obs_type = OBSTRUCTION_MAP.get(seg.obstruction_type, ObstructionType.UNKNOWN)
        action = RECOMMENDED_ACTIONS.get(
            (level.value, obs_type.value), seg.recommended_action
        )
        agencies = AGENCY_ROUTING.get(obs_type, [TargetAgency.AFAD])

        return AlertPayload(
            incident_id        = self._incident_id(seg),
            alert_level        = level,
            obstruction_type   = obs_type,
            location           = IncidentLocation(
                lat          = seg.centroid_latlon[0],
                lon          = seg.centroid_latlon[1],
                street_name  = seg.street_name,
                highway_type = seg.highway_type,
            ),
            recommended_action = action,
            target_agencies    = agencies,
            texture_score      = seg.texture_score,
            pixel_damage_ratio = seg.pixel_damage_ratio,
            passable_vehicles  = [v.value for v in seg.passable_vehicles],
            timestamp_utc      = now_utc,
            satellite_source   = self.satellite_source,
        )

    @staticmethod
    def _incident_id(seg) -> str:
        raw = f"{seg.centroid_latlon[0]:.5f}{seg.centroid_latlon[1]:.5f}{seg.obstruction_type}"
        return f"INC-{hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:8].upper()}"
