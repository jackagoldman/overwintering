"""
================================================================================
NASA FIRMS Hotspot Downloader — Overwinter Fire Detection (2023–2025)
================================================================================
Downloads VIIRS (SNPP, NOAA-20/J1, NOAA-21/J2) and MODIS active fire hotspots
from the NASA FIRMS API for a user-supplied AOI (SHP or GeoJSON).

Designed for detecting fires that overwintered between:
  • 2023 fire season → dormant winter 2023/2024 → 2024 fire season
  • 2024 fire season → dormant winter 2024/2025 → 2025 fire season

SETUP
-----
1. Get a free NASA FIRMS MAP KEY:
   https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Install dependencies:
   pip install requests geopandas shapely pandas pyproj fiona

USAGE
-----
  python firms_hotspot_downloader.py \
      --aoi path/to/your_zone.shp \
      --map-key YOUR_FIRMS_MAP_KEY \
      --out-dir ./firms_output

  python firms_hotspot_downloader.py \
      --aoi path/to/your_zone.geojson \
      --map-key YOUR_FIRMS_MAP_KEY \
      --out-dir ./firms_output \
      --sensors VIIRS_SNPP VIIRS_NOAA20 VIIRS_NOAA21 \
      --start-year 2023 \
      --end-year 2025

NOTES ON THE FIRMS API
-----------------------
The FIRMS area API accepts a bounding box (or country/region code).
For custom polygons, we derive the bounding box from your AOI and then
spatially clip the results to your exact polygon boundary.

The API endpoint pattern is:
  https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAYS}/{DATE}

  SOURCE options:
    VIIRS_SNPP_SP     → Suomi-NPP VIIRS (375m), standard processing
    VIIRS_NOAA20_SP   → NOAA-20 (J1) VIIRS (375m), standard processing
    VIIRS_NOAA21_SP   → NOAA-21 (J2) VIIRS (375m), standard processing
    MODIS_NRT         → MODIS Terra+Aqua (1km), near real-time
    MODIS_SP          → MODIS Terra+Aqua (1km), standard processing

  BBOX format: W,S,E,N (decimal degrees, WGS84)
  DAYS: 1–10 days per request (we paginate automatically)
  DATE: YYYY-MM-DD (start date of the DAYS window)

DATE STRATEGY FOR OVERWINTER FIRES
------------------------------------
We download the FULL year for each year (Jan–Dec) so you capture:
  - The start of fires in summer/fall 2023
  - The "gap" (winter) where surface hotspots disappear
  - Re-ignition signals in spring/summer 2024 in the same location
  - Repeat for 2024→2025 transition

================================================================================
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union

# Load .env file if present (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; set FIRMS_MAP_KEY in your shell environment instead

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DAYS_PER_REQUEST = 5         # Max days per single FIRMS API call
REQUEST_DELAY_SEC = 1.5        # Polite delay between requests
MAX_RETRIES = 3                # Retry attempts on HTTP errors

# Sensor source strings for the FIRMS API
SENSOR_SOURCES = {
    "VIIRS_SNPP":  "VIIRS_SNPP_SP",
    "VIIRS_NOAA20": "VIIRS_NOAA20_SP",   # J1
    "VIIRS_NOAA21": "VIIRS_NOAA21_SP",   # J2
    "MODIS":       "MODIS_SP",
}

# Availability dates — FIRMS may not have data before these dates
SENSOR_START_DATES = {
    "VIIRS_SNPP":   date(2012, 1, 20),
    "VIIRS_NOAA20": date(2018, 1, 1),
    "VIIRS_NOAA21": date(2023, 1, 1),    # J2 operational ~early 2023
    "MODIS":        date(2000, 11, 1),
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def load_aoi(aoi_path: str) -> gpd.GeoDataFrame:
    """Load a SHP or GeoJSON AOI, reproject to WGS84, and dissolve to one geometry."""
    log.info(f"Loading AOI: {aoi_path}")
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        log.warning("AOI has no CRS defined — assuming WGS84 (EPSG:4326).")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        log.info(f"Reprojecting AOI from {gdf.crs} to WGS84...")
        gdf = gdf.to_crs(epsg=4326)

    # Dissolve all features into one geometry for clipping
    dissolved = gdf.dissolve()
    log.info(f"AOI loaded: {len(gdf)} feature(s) dissolved into 1 polygon.")
    return dissolved


def get_bbox(gdf: gpd.GeoDataFrame) -> tuple:
    """Return (W, S, E, N) bounding box of the AOI."""
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    # Add a small buffer (0.1°) to catch edge hotspots
    buf = 0.1
    W = round(bounds[0] - buf, 4)
    S = round(bounds[1] - buf, 4)
    E = round(bounds[2] + buf, 4)
    N = round(bounds[3] + buf, 4)
    log.info(f"Bounding box (W,S,E,N): {W}, {S}, {E}, {N}")
    return W, S, E, N


def date_windows(start: date, end: date, window_days: int = DAYS_PER_REQUEST):
    """Generate (window_start, n_days) tuples covering [start, end]."""
    current = start
    while current <= end:
        remaining = (end - current).days + 1
        n = min(window_days, remaining)
        yield current, n
        current += timedelta(days=n)


def fetch_firms_csv(map_key: str, source: str, bbox: tuple,
                    start_date: date, n_days: int) -> pd.DataFrame | None:
    """Fetch a single FIRMS CSV window and return as DataFrame."""
    W, S, E, N = bbox
    bbox_str = f"{W},{S},{E},{N}"
    date_str = start_date.strftime("%Y-%m-%d")
    url = f"{FIRMS_BASE_URL}/{map_key}/{source}/{bbox_str}/{n_days}/{date_str}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                if resp.text.strip() == "" or resp.text.startswith("<?xml"):
                    return None  # No data in this window
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                return df if not df.empty else None
            elif resp.status_code == 429:
                wait = 30 * attempt
                log.warning(f"Rate limited. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
            elif resp.status_code == 400:
                log.error(f"Bad request for {url}: {resp.text[:200]}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code} on attempt {attempt}/{MAX_RETRIES}: {url}")
                time.sleep(5 * attempt)
        except requests.RequestException as e:
            log.warning(f"Request error on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(5 * attempt)

    log.error(f"Failed after {MAX_RETRIES} attempts: {url}")
    return None


def df_to_geodataframe(df: pd.DataFrame, sensor: str) -> gpd.GeoDataFrame | None:
    """Convert a FIRMS CSV DataFrame to a GeoDataFrame."""
    if df is None or df.empty:
        return None

    # Latitude/longitude column names vary slightly across sensors
    lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon", "long")), None)

    if lat_col is None or lon_col is None:
        log.warning(f"Could not find lat/lon columns in response. Columns: {list(df.columns)}")
        return None

    df = df.copy()
    df["sensor"] = sensor
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def clip_to_aoi(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatially clip hotspots to the exact AOI polygon."""
    aoi_geom = aoi.geometry.iloc[0]
    mask = gdf.geometry.within(aoi_geom) | gdf.geometry.intersects(aoi_geom)
    return gdf[mask].copy()


def download_sensor_year(map_key: str, sensor: str, source: str,
                          bbox: tuple, year: int,
                          aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """Download all hotspots for one sensor + one calendar year."""
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), date.today() - timedelta(days=1))

    # Check sensor availability
    sensor_start = SENSOR_START_DATES.get(sensor, date(2000, 1, 1))
    if end < sensor_start:
        log.info(f"  ↳ {sensor} not available for {year} (starts {sensor_start}). Skipping.")
        return None
    if start < sensor_start:
        start = sensor_start

    log.info(f"  Downloading {sensor} ({source}) for {year}: {start} → {end}")

    frames = []
    windows = list(date_windows(start, end))
    for i, (win_start, n_days) in enumerate(windows):
        log.info(f"    [{i+1}/{len(windows)}] {win_start} (+{n_days} days)")
        df = fetch_firms_csv(map_key, source, bbox, win_start, n_days)
        if df is not None:
            gdf_chunk = df_to_geodataframe(df, sensor)
            if gdf_chunk is not None and not gdf_chunk.empty:
                clipped = clip_to_aoi(gdf_chunk, aoi)
                if not clipped.empty:
                    frames.append(clipped)
                    log.info(f"      → {len(clipped)} hotspots within AOI")
        time.sleep(REQUEST_DELAY_SEC)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates()
        log.info(f"  ✓ {sensor} {year}: {len(combined)} total hotspots in AOI")
        return combined
    else:
        log.info(f"  ✗ {sensor} {year}: 0 hotspots found in AOI")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download NASA FIRMS hotspots for overwinter fire detection (2023–2025)"
    )
    parser.add_argument(
        "--aoi", required=True,
        help="Path to AOI file (.shp or .geojson)"
    )
    parser.add_argument(
        "--out-dir", default="./firms_output",
        help="Output directory for downloaded data (default: ./firms_output)"
    )
    parser.add_argument(
        "--sensors", nargs="+",
        default=["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"],
        choices=list(SENSOR_SOURCES.keys()),
        help="Sensors to download (default: all three VIIRS sensors)"
    )
    parser.add_argument(
        "--start-year", type=int, default=2023,
        help="First year to download (default: 2023)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2025,
        help="Last year to download (default: 2025)"
    )
    parser.add_argument(
        "--include-modis", action="store_true",
        help="Also download MODIS 1km data (useful as a cross-check)"
    )
    args = parser.parse_args()

    # ── Resolve MAP KEY from environment ──────────────────────────────────────
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        log.error(
            "No FIRMS MAP KEY found.\n"
            "  Option 1 — set it in your shell:\n"
            "    export FIRMS_MAP_KEY=your_key_here\n"
            "  Option 2 — create a .env file next to this script:\n"
            "    FIRMS_MAP_KEY=your_key_here\n"
            "  Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensors = args.sensors
    if args.include_modis and "MODIS" not in sensors:
        sensors = sensors + ["MODIS"]

    years = list(range(args.start_year, args.end_year + 1))

    # ── Load AOI ──────────────────────────────────────────────────────────────
    aoi = load_aoi(args.aoi)
    bbox = get_bbox(aoi)

    # ── Quick API key validation ───────────────────────────────────────────────
    log.info("Validating FIRMS API key...")
    test_url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/map_key_check"
        f"?map_key={map_key}&format=json"
    )
    try:
        r = requests.get(test_url, timeout=15)
        if r.status_code != 200:
            log.warning("Could not validate API key — proceeding anyway.")
        else:
            log.info(f"API key status: {r.text.strip()[:100]}")
    except Exception:
        log.warning("Key validation check skipped (network issue).")

    # ── Download loop ─────────────────────────────────────────────────────────
    all_data = []

    for sensor in sensors:
        source = SENSOR_SOURCES[sensor]
        log.info(f"\n{'='*60}")
        log.info(f"Sensor: {sensor}  ({source})")
        log.info(f"{'='*60}")

        for year in years:
            gdf = download_sensor_year(
                map_key=map_key,
                sensor=sensor,
                source=source,
                bbox=bbox,
                year=year,
                aoi=aoi,
            )
            if gdf is not None and not gdf.empty:
                gdf["year"] = year
                all_data.append(gdf)

                # Save per-sensor-per-year file
                stem = f"{sensor}_{year}"
                geojson_path = out_dir / f"{stem}.geojson"
                csv_path = out_dir / f"{stem}.csv"
                gdf.to_file(geojson_path, driver="GeoJSON")
                gdf.drop(columns="geometry").to_csv(csv_path, index=False)
                log.info(f"  Saved → {geojson_path}  ({len(gdf)} records)")

    # ── Merge and save combined output ────────────────────────────────────────
    if all_data:
        log.info(f"\n{'='*60}")
        log.info("Combining all sensors and years...")
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates()

        # Add a useful datetime column if ACQ_DATE / ACQ_TIME are present
        if "acq_date" in combined.columns:
            combined["acq_datetime"] = pd.to_datetime(
                combined["acq_date"].astype(str), errors="coerce"
            )

        combined_geojson = out_dir / "all_hotspots_2023_2025.geojson"
        combined_csv = out_dir / "all_hotspots_2023_2025.csv"
        combined.to_file(combined_geojson, driver="GeoJSON")
        combined.drop(columns="geometry").to_csv(combined_csv, index=False)

        log.info(f"\n✅ Done! Total hotspots: {len(combined)}")
        log.info(f"   Combined GeoJSON → {combined_geojson}")
        log.info(f"   Combined CSV     → {combined_csv}")

        # Summary table
        log.info("\nHotspot counts by sensor and year:")
        summary = combined.groupby(["sensor", "year"]).size().unstack(fill_value=0)
        print(summary.to_string())

    else:
        log.warning("\n⚠️  No hotspots were downloaded. Check your MAP KEY, AOI, and date range.")

    log.info("\nNext steps for overwinter fire analysis:")
    log.info("  1. Load all_hotspots_2023_2025.geojson into QGIS/ArcGIS/Python")
    log.info("  2. Cluster hotspots spatially (e.g., DBSCAN) by location")
    log.info("  3. For each cluster, check if detections span BOTH sides of a winter gap")
    log.info("     e.g., hotspots in Aug–Oct 2023 AND Mar–May 2024 within <500m")
    log.info("  4. Cross-reference with soil temperature / snow cover data")
    log.info("     (MODIS MOD11A1 LST or ERA5 soil temp) to confirm dormant phase")


if __name__ == "__main__":
    main()