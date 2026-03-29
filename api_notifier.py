# ============================================================
# api_notifier.py - AFAD Kurumsal Bildirim ve Raporlama
# TUA Astro Hackathon 2026
# ============================================================
"""
AFAD (Afet ve Acil Durum Yönetimi Başkanlığı) için kurumsal
bildirim ve raporlama modülü.

Özellikler:
  - Standart AFAD JSON payload formatı
  - Tarayıcıdan PDF olarak bastırılabilir HTML rapor
  - AFAD API senkronizasyon simülasyonu
  - Incident ID ve checksum ile izlenebilirlik

Gerçek AFAD Entegrasyonu:
  simulate_afad_sync() metodundaki HTTP POST stub'ı gerçek
  endpoint ile değiştirilerek canlı entegrasyon sağlanabilir.
  Endpoint: https://api.afad.gov.tr/v1/disaster-reports
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ============================================================
#  DATACLASSES
# ============================================================

@dataclass
class AFADRoadEntry:
    road_name: str
    osm_id: str
    status: str          # "BLOCKED" | "DAMAGED" | "OPEN"
    damage_ratio: float
    damage_percentage: str
    length_m: float
    coords: List[List[float]]  # [[lat, lon], ...]


@dataclass
class AFADRouteEntry:
    route_id: int
    label: str
    total_length_m: float
    total_time_min: float
    passes_damage: bool
    coords: List[List[float]]


@dataclass
class AFADPayload:
    incident_id: str
    disaster_type: str
    disaster_type_tr: str
    timestamp: str
    location_name: str
    location_coords: Dict          # {"lat": float, "lon": float}
    closed_roads: List[AFADRoadEntry]
    alternative_routes: List[AFADRouteEntry]
    estimated_damage_level: Dict
    satellite_source: str          # "İMECE" | "GÖKTÜRK-1" | "Demo"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ============================================================
#  NOTIFICATION BUILDER
# ============================================================

class NotificationBuilder:
    """
    AFAD için standart bildirim paketi oluşturan sınıf.

    Kullanım:
        builder = NotificationBuilder()
        payload = builder.build_from_analysis(
            report=rpt,
            road_statuses=road_damage,
            route_primary=route_primary,
            route_alternative=route_alt,
        )
        html = builder.to_html_report(payload)
        result = builder.simulate_afad_sync(payload)
    """

    DISASTER_TYPES_TR = {
        "earthquake": "Deprem",
        "flood": "Sel / Su Taşkını",
        "fire": "Yangın",
        "landslide": "Heyelan",
        "explosion": "Patlama",
        "other": "Diğer",
    }

    # ────────────────────────────────────────────────────────
    def build_from_analysis(
        self,
        report,
        road_statuses: Dict,
        route_primary: Optional[Dict],
        route_alternative: Optional[Dict],
        disaster_type: str = "earthquake",
        location_name: str = "Afet Bölgesi",
        center_coords: Tuple[float, float] = (37.75, 37.01),
        satellite_source: str = "İMECE",
    ) -> AFADPayload:
        """
        Analiz sonuçlarından standart AFAD payload'ı oluşturur.

        Args:
            report: DamageEngine raporu (get_executive_summary() desteklemeli).
            road_statuses: analyze_road_damage() çıktısı.
            route_primary: find_safe_route() çıktısı.
            route_alternative: find_alternative_route() çıktısı.
            disaster_type: Afet türü (İngilizce anahtar).
            location_name: Konum adı (ör. "Hatay").
            center_coords: (lat, lon) koordinatı.
            satellite_source: Kullanılan uydu.

        Returns:
            AFADPayload nesnesi.
        """
        incident_id = (
            f"ASTRO-{datetime.now():%Y%m%d%H%M}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        # Executive summary
        es: Dict = {}
        if hasattr(report, "get_executive_summary"):
            try:
                es = report.get_executive_summary()
            except Exception:
                es = {}

        # Kapalı / hasarlı yollar
        closed_roads: List[AFADRoadEntry] = []
        for osm_id, data in road_statuses.items():
            status = data.get("status", "OPEN")
            if status in ("BLOCKED", "DAMAGED"):
                dmg_ratio = data.get("damage_ratio", 0.0)
                closed_roads.append(AFADRoadEntry(
                    road_name=data.get("road_name", "İsimsiz Yol"),
                    osm_id=osm_id,
                    status=status,
                    damage_ratio=dmg_ratio,
                    damage_percentage=f"{dmg_ratio * 100:.1f}%",
                    length_m=data.get("length_m", 0.0),
                    coords=[[c[0], c[1]] for c in data.get("geometry_coords", [])],
                ))

        # Güvenli rotalar
        alternative_routes: List[AFADRouteEntry] = []
        for route_id, label, route in [
            (1, "Birincil Güvenli Rota", route_primary),
            (2, "Alternatif Güvenli Rota", route_alternative),
        ]:
            if route:
                coords_raw = route.get("coordinates", [])
                alternative_routes.append(AFADRouteEntry(
                    route_id=route_id,
                    label=label,
                    total_length_m=route.get("total_length_m", 0.0),
                    total_time_min=round(route.get("total_time_s", 0.0) / 60.0, 1),
                    passes_damage=bool(route.get("passes_damage", False)),
                    coords=[[c[0], c[1]] for c in coords_raw],
                ))

        # Tahminî hasar seviyesi
        blocked_list  = [v for v in road_statuses.values() if v.get("status") == "BLOCKED"]
        damaged_list  = [v for v in road_statuses.values() if v.get("status") == "DAMAGED"]
        total_blocked_length = sum(v.get("length_m", 0) for v in blocked_list)

        estimated_damage_level = {
            "ssim_score":              es.get("ssim_score", 0.0),
            "destroyed_buildings":     es.get("destroyed_buildings", 0),
            "heavy_damage_buildings":  es.get("heavy_damage_buildings", 0),
            "damage_area_m2":          es.get("damage_area_m2", 0.0),
            "damage_ratio_percent":    round(es.get("damage_ratio", 0.0) * 100, 2),
            "total_blocked_roads":     len(blocked_list),
            "total_damaged_roads":     len(damaged_list),
            "total_blocked_length_m":  round(total_blocked_length, 1),
        }

        payload = AFADPayload(
            incident_id=incident_id,
            disaster_type=disaster_type,
            disaster_type_tr=self.DISASTER_TYPES_TR.get(disaster_type, "Diğer"),
            timestamp=timestamp,
            location_name=location_name,
            location_coords={"lat": center_coords[0], "lon": center_coords[1]},
            closed_roads=closed_roads,
            alternative_routes=alternative_routes,
            estimated_damage_level=estimated_damage_level,
            satellite_source=satellite_source,
            metadata={
                "system": "ASTRO-RESQ",
                "version": "2.0",
                "tua_hackathon": "2026",
                "generated_by": "DamageEngine + SafeRouteOptimizer",
            },
        )

        logger.info(
            f"AFAD payload oluşturuldu | incident_id={incident_id} | "
            f"kapalı_yol={len(closed_roads)} | rota={len(alternative_routes)}"
        )
        return payload

    # ────────────────────────────────────────────────────────
    def to_html_report(self, payload: AFADPayload) -> str:
        """
        Kurumsal kalitede, tarayıcıdan PDF olarak bastırılabilir
        HTML rapor üretir.
        """
        dl = payload.estimated_damage_level
        now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

        # Yol tablosu satırları
        road_rows_html = ""
        for road in payload.closed_roads:
            badge_color = "#e63946" if road.status == "BLOCKED" else "#f77f00"
            badge_label = "KAPALI" if road.status == "BLOCKED" else "HASARLI"
            road_rows_html += f"""
            <tr>
                <td>{road.road_name}</td>
                <td><span class="badge" style="background:{badge_color}">{badge_label}</span></td>
                <td>{road.damage_percentage}</td>
                <td>{road.length_m:.0f} m</td>
            </tr>"""

        if not road_rows_html:
            road_rows_html = '<tr><td colspan="4" style="text-align:center;opacity:0.5">Kapalı yol tespit edilmedi</td></tr>'

        # Rota tablosu satırları
        route_rows_html = ""
        for route in payload.alternative_routes:
            status_icon = "⚠️ Dikkatli" if route.passes_damage else "✅ Güvenli"
            route_rows_html += f"""
            <tr>
                <td><b>{route.label}</b></td>
                <td>{route.total_length_m:.0f} m</td>
                <td>{route.total_time_min:.1f} dk</td>
                <td>{status_icon}</td>
            </tr>"""

        if not route_rows_html:
            route_rows_html = '<tr><td colspan="4" style="text-align:center;opacity:0.5">Rota hesaplanamadı</td></tr>'

        html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AFAD Afet Bildirimi — {payload.incident_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    font-size: 14px;
  }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}

  /* ── HEADER ── */
  .header {{
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
    color: #fff;
    padding: 32px 36px;
    border-radius: 14px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }}
  .header::after {{
    content: "🛰️";
    position: absolute;
    right: 32px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.12;
  }}
  .header .org {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 3px; opacity: 0.7; margin-bottom: 8px; }}
  .header h1 {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 12px; }}
  .header .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 8px; margin-top: 16px; }}
  .header .meta-item {{ background: rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 14px; }}
  .header .meta-item .lbl {{ font-size: 0.68rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 1px; }}
  .header .meta-item .val {{ font-size: 0.92rem; font-weight: 600; margin-top: 2px; }}

  /* ── SECTION CARDS ── */
  .section {{
    background: #fff;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid rgba(0,0,0,0.06);
  }}
  .section h2 {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #302b63;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f2f5;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  /* ── KPI CARDS ── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
  }}
  .kpi-card {{
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid transparent;
  }}
  .kpi-card .kpi-val {{ font-size: 2rem; font-weight: 800; line-height: 1; margin: 4px 0; font-variant-numeric: tabular-nums; }}
  .kpi-card .kpi-lbl {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.75; }}
  .kpi-red    {{ background: #fff5f5; border-color: #ffd6d6; }}
  .kpi-red    .kpi-val {{ color: #e63946; }}
  .kpi-orange {{ background: #fff8f0; border-color: #ffddb8; }}
  .kpi-orange .kpi-val {{ color: #f77f00; }}
  .kpi-blue   {{ background: #f0f7ff; border-color: #bfdbff; }}
  .kpi-blue   .kpi-val {{ color: #0077b6; }}
  .kpi-green  {{ background: #f0fff8; border-color: #b6f2da; }}
  .kpi-green  .kpi-val {{ color: #06843a; }}
  .kpi-purple {{ background: #f7f0ff; border-color: #d8b4fe; }}
  .kpi-purple .kpi-val {{ color: #6d28d9; }}

  /* ── TABLES ── */
  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    background: #f7f8fc;
    padding: 10px 14px;
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7280;
    border-bottom: 2px solid #e5e7eb;
  }}
  td {{
    padding: 10px 14px;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.88rem;
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f9fafb; }}

  /* ── BADGES ── */
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.5px;
  }}

  /* ── SYSTEM INFO ── */
  .sys-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .sys-row {{ display: flex; justify-content: space-between; padding: 8px 12px; background: #f7f8fc; border-radius: 6px; }}
  .sys-row .sk {{ font-size: 0.8rem; color: #6b7280; font-weight: 500; }}
  .sys-row .sv {{ font-size: 0.8rem; font-weight: 700; color: #1a1a2e; font-family: 'Courier New', monospace; }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center;
    padding: 20px;
    font-size: 0.78rem;
    color: #9ca3af;
    border-top: 1px solid #e5e7eb;
    margin-top: 8px;
  }}
  .footer .print-hint {{
    display: inline-block;
    background: #f0f7ff;
    border: 1px solid #bfdbff;
    border-radius: 6px;
    padding: 8px 16px;
    color: #0077b6;
    font-weight: 600;
    margin-top: 8px;
  }}

  /* ── PRINT ── */
  @media print {{
    body {{ background: #fff; }}
    .page {{ max-width: 100%; padding: 0; }}
    .section {{ box-shadow: none; border: 1px solid #ddd; break-inside: avoid; }}
    .header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .kpi-card {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .footer .print-hint {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div class="org">T.C. İçişleri Bakanlığı • AFAD Afet ve Acil Durum Yönetimi Başkanlığı</div>
    <h1>🚨 Afet Erken Müdahale Raporu</h1>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="lbl">Olay ID</div>
        <div class="val">{payload.incident_id}</div>
      </div>
      <div class="meta-item">
        <div class="lbl">Afet Türü</div>
        <div class="val">{payload.disaster_type_tr}</div>
      </div>
      <div class="meta-item">
        <div class="lbl">Konum</div>
        <div class="val">{payload.location_name} ({payload.location_coords['lat']:.4f}, {payload.location_coords['lon']:.4f})</div>
      </div>
      <div class="meta-item">
        <div class="lbl">Uydu Kaynağı</div>
        <div class="val">🛰️ {payload.satellite_source}</div>
      </div>
      <div class="meta-item">
        <div class="lbl">Rapor Tarihi</div>
        <div class="val">{now_str}</div>
      </div>
      <div class="meta-item">
        <div class="lbl">Sistem</div>
        <div class="val">ASTRO-RESQ v2.0</div>
      </div>
    </div>
  </div>

  <!-- KPI CARDS -->
  <div class="section">
    <h2>📊 Hasar Özet Göstergeleri</h2>
    <div class="kpi-grid">
      <div class="kpi-card kpi-red">
        <div class="kpi-lbl">Yıkık Bina</div>
        <div class="kpi-val">{dl.get('destroyed_buildings', 0)}</div>
      </div>
      <div class="kpi-card kpi-orange">
        <div class="kpi-lbl">Ağır Hasarlı Bina</div>
        <div class="kpi-val">{dl.get('heavy_damage_buildings', 0)}</div>
      </div>
      <div class="kpi-card kpi-red">
        <div class="kpi-lbl">Kapalı Yol Segmenti</div>
        <div class="kpi-val">{dl.get('total_blocked_roads', 0)}</div>
      </div>
      <div class="kpi-card kpi-orange">
        <div class="kpi-lbl">Kapalı Yol Uzunluğu</div>
        <div class="kpi-val">{dl.get('total_blocked_length_m', 0):.0f}<span style="font-size:1rem;font-weight:400"> m</span></div>
      </div>
      <div class="kpi-card kpi-purple">
        <div class="kpi-lbl">Hasar Oranı</div>
        <div class="kpi-val">{dl.get('damage_ratio_percent', 0):.1f}<span style="font-size:1rem;font-weight:400">%</span></div>
      </div>
      <div class="kpi-card kpi-green">
        <div class="kpi-lbl">Hasar Alanı</div>
        <div class="kpi-val">{dl.get('damage_area_m2', 0):,.0f}<span style="font-size:0.85rem;font-weight:400"> m²</span></div>
      </div>
    </div>
  </div>

  <!-- KAPALI YOLLAR -->
  <div class="section">
    <h2>🚧 Kapalı ve Hasarlı Yol Segmentleri ({len(payload.closed_roads)} adet)</h2>
    <table>
      <thead>
        <tr>
          <th>Yol Adı</th>
          <th>Durum</th>
          <th>Hasar %</th>
          <th>Uzunluk</th>
        </tr>
      </thead>
      <tbody>
        {road_rows_html}
      </tbody>
    </table>
  </div>

  <!-- GÜVENLİ ROTALAR -->
  <div class="section">
    <h2>🗺️ Hesaplanan Güvenli Rotalar</h2>
    <table>
      <thead>
        <tr>
          <th>Rota</th>
          <th>Mesafe</th>
          <th>Süre</th>
          <th>Durum</th>
        </tr>
      </thead>
      <tbody>
        {route_rows_html}
      </tbody>
    </table>
  </div>

  <!-- SİSTEM BİLGİSİ -->
  <div class="section">
    <h2>⚙️ Sistem Bilgileri</h2>
    <div class="sys-grid">
      <div class="sys-row"><span class="sk">Sistem</span><span class="sv">{payload.metadata.get('system','—')}</span></div>
      <div class="sys-row"><span class="sk">Versiyon</span><span class="sv">{payload.metadata.get('version','—')}</span></div>
      <div class="sys-row"><span class="sk">Hackathon</span><span class="sv">TUA Astro {payload.metadata.get('tua_hackathon','—')}</span></div>
      <div class="sys-row"><span class="sk">Modüller</span><span class="sv">{payload.metadata.get('generated_by','—')}</span></div>
      <div class="sys-row"><span class="sk">SSIM Skoru</span><span class="sv">{dl.get('ssim_score',0):.4f}</span></div>
      <div class="sys-row"><span class="sk">Uydu Kaynağı</span><span class="sv">{payload.satellite_source}</span></div>
      <div class="sys-row"><span class="sk">UTC Zaman Damgası</span><span class="sv">{payload.timestamp}</span></div>
      <div class="sys-row"><span class="sk">Olay ID</span><span class="sv">{payload.incident_id}</span></div>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <p>Bu rapor <b>ASTRO-RESQ</b> karar destek sistemi tarafından otomatik olarak oluşturulmuştur.</p>
    <p>T.C. İçişleri Bakanlığı AFAD için | TUA Astro Hackathon 2026</p>
    <div class="print-hint">💡 PDF kaydetmek için: Tarayıcıda <kbd>Ctrl+P</kbd> → Yazıcı: "PDF'e kaydet" → Yazdır</div>
  </div>

</div>
</body>
</html>"""
        return html

    # ────────────────────────────────────────────────────────
    def simulate_afad_sync(
        self,
        payload: AFADPayload,
        endpoint_url: str = "https://api.afad.gov.tr/v1/disaster-reports",
    ) -> Dict:
        """
        AFAD API senkronizasyonunu simüle eder.

        Gerçek Entegrasyon Notu:
        ─────────────────────────────────────────────────────
        Bu metod şu an bir simülasyon döndürmektedir. Canlı
        entegrasyon için aşağıdaki stub'ı etkinleştirin:

            import requests
            response = requests.post(
                endpoint_url,
                json=payload.to_dict(),
                headers={
                    "Authorization": "Bearer <AFAD_API_TOKEN>",
                    "Content-Type": "application/json",
                    "X-ASTRO-RESQ-Version": "2.0",
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        Endpoint: https://api.afad.gov.tr/v1/disaster-reports
        """
        logger.info(
            f"AFAD senkronizasyonu başlatıldı | "
            f"incident={payload.incident_id} | endpoint={endpoint_url}"
        )

        try:
            payload_json = payload.to_json()
            checksum = hashlib.md5(payload_json.encode()).hexdigest()
            afad_tracking_no = (
                f"AFAD-{datetime.now():%Y%m%d}-{checksum[:8].upper()}"
            )

            result = {
                "status": "success",
                "incident_id": payload.incident_id,
                "afad_tracking_no": afad_tracking_no,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Afet bildirimi başarıyla AFAD sistemine iletildi.",
                "endpoint": endpoint_url,
                "payload_size_bytes": len(payload_json.encode("utf-8")),
            }

            logger.success(
                f"AFAD sync başarılı | takip_no={afad_tracking_no} | "
                f"boyut={result['payload_size_bytes']} byte"
            )
            return result

        except Exception as e:
            logger.error(f"AFAD sync hatası: {e}")
            return {
                "status": "error",
                "error": str(e),
                "incident_id": payload.incident_id,
            }
