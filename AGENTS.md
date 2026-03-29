# AGENTS.md

## Run
```bash
source .venv/bin/activate
streamlit run main.py
```
No test suite. No linter. Validate manually in browser.

---

## Must-Follow Constraints

### Streamlit
- `@st.cache_data` **must** be at module top-level — never inside a `with` block, tab, or function.
- Every `st_folium()` call **must** include `returned_objects=[]` — omitting it causes full page rerender on map click.
- Folium tile URL `attr` param **must** be `" "` (one space minimum) — empty string crashes the map.
- Use `attribution_control=False` on the `folium.Map(...)` call to hide attribution UI.

### Coordinate Order
- **Folium / internal storage / UI:** `(lat, lon)`
- **OSRM response geometry:** `[lon, lat]` — must swap before passing to Folium.
- **CEMS GeoJSON:** `[lon, lat]` — same swap required.

### Secrets
- Read via `st.secrets.get("service", {}).get("api_key", "")` — never hardcode.
- `.streamlit/secrets.toml` is in `.gitignore` — do not move or rename.

### AI Vision Backend
- Class: `AIVisionAnalyzer(api_key, backend="gemini")`
- Reference models as `self.GEMINI_MODEL` / `self.ANTHROPIC_MODEL` — `self.MODEL` does not exist.

### Random Seed
- `np.random.seed(42)` and `_rnd.seed(42)` are intentional — damage simulation must be reproducible for demo. Do not change.

---

## Repo-Specific Conventions

### Imports
- `cv2`, `folium`, `streamlit_folium`, `osmnx`, `networkx` are **lazy-imported inside functions**, not at top level — keep it that way to avoid slow startup.
- Pattern: `from module import Class as _Alias` inside the tab block.

### Session State Keys (load-bearing — do not rename)
`report`, `img_before`, `img_after`, `meta_before`, `meta_after`, `cems_features`, `cems_is_live`, `ai_vision_result`, `weather_decision`, `loj_route_pts`, `loj_inter_city`, `loj_cities`

### Tab Variables
`_tab_loj`, `_tab_met`, `_tab_seis`, `_tab_img`, `_tab_hav`, `_tab_acil` — defined once at the `st.tabs()` call, referenced throughout `main.py`.

### Configuration
Four frozen dataclass singletons from `config.py`: `app_config`, `cv_config`, `map_config`, `routing_config`. Do not instantiate new instances.

### Map Colors (use exactly)
| Meaning | Hex |
|---------|-----|
| Suggested route | `#4361ee` |
| Damaged road | `#f77f00` |
| Closed road | `#e63946` |
| Open road | `#06d6a0` |

---

## Known Gotchas

- **Copernicus WFS returns 404** for most layer types — this is expected. Mock fallback is correct behavior. Do not attempt to fix the WFS endpoint.
- **`self.blocked_edges`** must be initialized in `SafeRouteOptimizer.__init__` alongside `self.damaged_edges` — missing init causes `AttributeError` at runtime.
- **Planet key write** — user input must be validated with `re.fullmatch(r'[A-Za-z0-9\-_]{20,80}', key)` before writing to `secrets.toml` to prevent TOML injection.
- **`st.cache_data.clear()`** nukes all function caches simultaneously — if adding a targeted refresh, use a `_bust: int = 0` parameter instead.
- OSRM routing uses `https://router.project-osrm.org` — do not revert to `http://`.
