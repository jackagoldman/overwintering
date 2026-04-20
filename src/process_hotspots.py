# this code is for processing hotspot data
# There are 4 hotspot data sources across 3 years (2023-2025) that we want to use:
#    DL_FIRE_M-C61_xxxx if you requested MODIS data (C61 stands for MODIS Collection 6.1), or
#    DL_FIRE_J1V-C2_xxx if you requested VIIRS 375m data from  NOAA-20(JPSS-1)
#    DL_FIRE_J2V-C2_xxx if you requested VIIRS 375m data from  NOAA-21(JPSS-2)
#    DL_FIRE_SV-C2_xxxx if you requested VIIRS 375m data from S-NPP
# all data is archive data, so processed using a 3-month lag except for the J2V-C2 data which is near real-time (nrt) data, so processed using a 1-day lag.

# Workflow
# 1. data needs to be clipped to the 3005 zone and merged together into a single dataset (with a column for the data source) - this will be the `hotspots_clipped` dataset
# 2. data will need to be split into 2023, 2024 and 2025 datasets
# 3. we will make a grid of 0.25 degree cells and assign a `first_time` variable to each cell based on the earliest lightning strike in that cell (product already available as `cg_first_occurrence_2024.nc`), and then we will convert that grid to a geodataframe with polygons for each cell
# 4. we will then spatially join the hotspots to the grid to get the `first_time` variable for each hotspot, and then we can calculate the time difference between the hotspot time and the first lightning time in that cell (2024 only)

# files:
#  root: ../data/raw/
#  - "2023-24_Hotspots/DL_FIRE_J1V-C2_666396/fire_archive_J1V-C2_666396.shp"
#  - "2023-24_Hotspots/DL_FIRE_J2V-C2_666397/fire_nrt_J2V-C2_666397.shp"
#  - "2023-24_Hotspots/DL_FIRE_M-C61_666395/fire_archive_M-C61_666395.shp"
#  - "2023-24_Hotspots/DL_FIRE_SV-C2_666398/fire_archive_SV-C2_666398.shp"
#  - "2024-25_Hotspots/DL_FIRE_J1V-C2_666373/fire_archive_J1V-C2_666373.shp"
#  - "2024-25_Hotspots/DL_FIRE_J2V-C2_666374/fire_nrt_J2V-C2_666374.shp"
#  - "2024-25_Hotspots/DL_FIRE_M-C61_666372/fire_archive_M-C61_666372.shp"
#  - "2024-25_Hotspots/DL_FIRE_SV-C2_666375/fire_archive_SV-C2_666375.shp"

# study zone
# - data/processed_study_zones/study_zones_3005.geojson

# lighnining data (2024 first occurrence) -
# - data/processed_lightning/cg_first_occurrence_2024.nc

# output files:
# - clipped and merged hotspot dataset: data/processed_hotspots/hotspots_clipped_all.shp and .geojson
# - clipped and merged hotspot datasets for each year: data/processed_hotspots/hotspots_clipped_2023.shp/.geojson, data/processed_hotspots/hotspots_clipped_2024.shp/.geojson, data/processed_hotspots/hotspots_clipped_2025.shp/.geojson
# - merged with lightning data (2024 only): data/processed_hotspots/hotspots_clipped_with_lightning_2024.shp/.geojson

# ACQ_DATE is the date of the hotspot, and it is in the format YYYYMMDD, so we will need to convert it to a datetime object and then filter by year to split into 2023, 2024 and 2025 datasets.

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from shapely.geometry import box

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed_hotspots"
STUDY_ZONE_PATH = ROOT / "data" / "processed_study_zones" / "study_zones_3005.geojson"
LIGHTNING_NC_PATH = ROOT / "data" / "processed_lightning" / "cg_first_occurrence_2024.nc"

PROCESSED.mkdir(exist_ok=True)

# (file path, sensor source label)
HOTSPOT_FILES = [
    (RAW / "2023-24_Hotspots/DL_FIRE_J1V-C2_666396/fire_archive_J1V-C2_666396.shp", "J1V-C2"),
    (RAW / "2023-24_Hotspots/DL_FIRE_J2V-C2_666397/fire_nrt_J2V-C2_666397.shp",     "J2V-C2"),
    (RAW / "2023-24_Hotspots/DL_FIRE_M-C61_666395/fire_archive_M-C61_666395.shp",    "M-C61"),
    (RAW / "2023-24_Hotspots/DL_FIRE_SV-C2_666398/fire_archive_SV-C2_666398.shp",    "SV-C2"),
    (RAW / "2024-25_Hotspots/DL_FIRE_J1V-C2_666373/fire_archive_J1V-C2_666373.shp", "J1V-C2"),
    (RAW / "2024-25_Hotspots/DL_FIRE_J2V-C2_666374/fire_nrt_J2V-C2_666374.shp",     "J2V-C2"),
    (RAW / "2024-25_Hotspots/DL_FIRE_M-C61_666372/fire_archive_M-C61_666372.shp",    "M-C61"),
    (RAW / "2024-25_Hotspots/DL_FIRE_SV-C2_666375/fire_archive_SV-C2_666375.shp",    "SV-C2"),
]


def save(gdf: gpd.GeoDataFrame, stem: str) -> None:
    """Save a GeoDataFrame as both .shp and .geojson."""
    shp_path = PROCESSED / f"{stem}.shp"
    gjson_path = PROCESSED / f"{stem}.geojson"
    gdf.to_file(shp_path)
    gdf.to_file(gjson_path, driver="GeoJSON")
    print(f"  saved {shp_path.name} and {gjson_path.name} ({len(gdf)} records)")


# ---------------------------------------------------------------------------
# Step 1: load, clip to study zone, merge with source column
# ---------------------------------------------------------------------------
print("Step 1: loading and clipping hotspots...")

study_zone_3005 = gpd.read_file(STUDY_ZONE_PATH)
study_zone_wgs84 = study_zone_3005.to_crs(epsg=4326)
clip_geom_wgs84 = study_zone_wgs84.union_all()

gdfs = []
for path, source in HOTSPOT_FILES:
    gdf = gpd.read_file(path)
    clipped = gdf.clip(clip_geom_wgs84)
    clipped = clipped.copy()
    clipped["source"] = source
    gdfs.append(clipped)
    print(f"  {path.name}: {len(clipped)} hotspots after clip")

hotspots_clipped = pd.concat(gdfs, ignore_index=True)
hotspots_clipped = gpd.GeoDataFrame(hotspots_clipped, crs="EPSG:4326")

# Parse ACQ_DATE to datetime (handles both 'YYYY-MM-DD' and 'YYYYMMDD' formats)
hotspots_clipped["ACQ_DATE"] = pd.to_datetime(
    hotspots_clipped["ACQ_DATE"].astype(str)
)
hotspots_clipped["year"] = hotspots_clipped["ACQ_DATE"].dt.year

# Reproject to EPSG:3005 to match study zone
hotspots_clipped = hotspots_clipped.to_crs(epsg=3005)

print(f"  total hotspots after merge: {len(hotspots_clipped)}")
save(hotspots_clipped, "hotspots_clipped_all")

# ---------------------------------------------------------------------------
# Step 2: split by year
# ---------------------------------------------------------------------------
print("Step 2: splitting by year...")

for year in [2023, 2024, 2025]:
    subset = hotspots_clipped[hotspots_clipped["year"] == year].copy()
    print(f"  {year}: {len(subset)} hotspots")
    save(subset, f"hotspots_clipped_{year}")

# ---------------------------------------------------------------------------
# Step 3: build lightning grid from cg_first_occurrence_2024.nc
# ---------------------------------------------------------------------------
print("Step 3: building lightning first-occurrence grid...")

ds = xr.open_dataset(LIGHTNING_NC_PATH)
first_time_da = ds["__xarray_dataarray_variable__"]

lats = first_time_da.lat.values
lons = first_time_da.lon.values

# Cell half-width derived from coordinate spacing
dlat = abs(float(lats[1] - lats[0])) / 2
dlon = abs(float(lons[1] - lons[0])) / 2

rows = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        t = first_time_da.values[i, j]
        if pd.isnull(t):
            continue
        cell_box = box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        rows.append({"geometry": cell_box, "first_time": pd.Timestamp(t), "grid_lat": lat, "grid_lon": lon})

lightning_grid = gpd.GeoDataFrame(rows, crs="EPSG:4326")
print(f"  lightning grid: {len(lightning_grid)} non-null cells")

# ---------------------------------------------------------------------------
# Step 4: spatial join hotspots_2024 to lightning grid, compute time delta
# ---------------------------------------------------------------------------
print("Step 4: joining 2024 hotspots to lightning grid...")

hotspots_2024 = hotspots_clipped[hotspots_clipped["year"] == 2024].copy()

# Reproject grid to 3005 to match hotspots
lightning_grid_3005 = lightning_grid.to_crs(epsg=3005)

joined = gpd.sjoin(
    hotspots_2024,
    lightning_grid_3005[["geometry", "first_time", "grid_lat", "grid_lon"]],
    how="left",
    predicate="within",
)

# Build a UTC datetime for each hotspot using ACQ_DATE + ACQ_TIME (HHMM integer)
# ACQ_TIME may not be present in all datasets; handle gracefully
if "ACQ_TIME" in joined.columns:
    acq_time_str = joined["ACQ_TIME"].fillna(0).astype(int).astype(str).str.zfill(4)
    hours = acq_time_str.str[:2].astype(int)
    minutes = acq_time_str.str[2:].astype(int)
    joined["hotspot_datetime"] = joined["ACQ_DATE"] + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
else:
    joined["hotspot_datetime"] = joined["ACQ_DATE"]

# Positive delta means hotspot detected after first lightning strike in cell
joined["hours_since_lightning"] = (
    joined["hotspot_datetime"] - joined["first_time"]
).dt.total_seconds() / 3600

print(f"  joined 2024 hotspots: {len(joined)} records")
print(f"  hotspots with a matched lightning cell: {joined['first_time'].notna().sum()}")

save(joined, "hotspots_clipped_with_lightning_2024")

print("Done.")
