# =============================================================================
# OVERWINTERING FIRE DETECTION ANALYSIS
# =============================================================================
# Detects fires that overwintered (zombie fires) in northeastern BC boreal
# plains between 2023-2025 using VIIRS FIRMS hotspots and BC fire perimeters.
#
# Algorithm based on Scholten et al.:
#   - 1000m buffer around previous year fire perimeters
#   - Spring hotspot detection after snowmelt onset (SDD)
#   - Spring window: SDD to SDD + 50 days
#   - Maximum travel distance: 1000m between last fall and first spring hotspot
#
# Study region: Fort St. John and Fort Nelson Fire Zones, northeastern BC
# =============================================================================

# --- imports -----------------------------------------------------------------
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import euclidean
from pyproj import Transformer
from rasterstats import zonal_stats
import rasterio



# =============================================================================
# CONSTANTS
# =============================================================================

# coordinate transformer: WGS84 -> BC Albers for distance calculations
TRANSFORMER = Transformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)

# buffer distances for sensitivity analysis (Scholten et al. use 1000m)
BUFFER_DISTANCES = [500, 1000, 2000]

# fall hotspot window: after Aug 1 (DOY 213) through end of year
FALL_DOY_START = 213

# Scholten et al. thresholds
MAX_HOTSPOT_DISTANCE_M = 1000   # max travel distance between fall and spring hotspot
SDD_MAX_DAYS           = 50     # max days after snowmelt for spring detection

# zone-level SDD lookup (from MODIS fractional snow cover, DOY)
SDD_LOOKUP = {
    'Fort St. John Fire Zone': {2024: 135.71, 2025: 140.63},
    'Fort Nelson Fire Zone':   {2024: 165.14, 2025: 167.72}
}

# SDD raster paths
SDD_RASTERS = {
    2024: '../data/sdd/sdd_2024.tif',
    2025: '../data/sdd/sdd_2025.tif'
}

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

# --- fire perimeters (2023-2025) ---
fires = gpd.read_file('data/analysis/all_fires_processed.geojson')
fires['fire_year'] = pd.to_numeric(fires['fire_year'], errors='coerce')

print(f"Fires loaded: {len(fires)}")
print(f"Years: {sorted(fires['fire_year'].unique())}")
print(f"Zones: {fires['fire_zone'].unique()}")
print(f"CRS: {fires.crs}")

# --- buffered fire perimeters with per-buffer SDD ---
# buffers at 500m, 1000m, 2000m around 2023 and 2024 perimeters
# SDD (mean/min/max) extracted per buffer using extract_sdd_stats()
buffered_dfs = {
    500:  gpd.read_file('data/analysis/fires_buffer_500m_with_sdd.geojson'),
    1000: gpd.read_file('data/analysis/fires_buffer_1000m_with_sdd.geojson'),
    2000: gpd.read_file('data/analysis/fires_buffer_2000m_with_sdd.geojson')
}

for dist in BUFFER_DISTANCES:
    sdd_cols = [c for c in buffered_dfs[dist].columns if 'sdd' in c]
    print(f"Buffer {dist}m: {len(buffered_dfs[dist])} fires | SDD cols: {sdd_cols}")

# --- VIIRS FIRMS hotspots ---
# pre-filtered: nighttime only (daynight == 'N'),
#               confidence != low, scan/track <= 0.5
hotspots = gpd.read_file('firms_output/all_hotspots_2023_2025.geojson')

# ensure datetime, DOY and year columns exist
hotspots['datetime'] = pd.to_datetime(hotspots['acq_date'])
hotspots['doy']      = hotspots['datetime'].dt.dayofyear
hotspots['year']     = hotspots['datetime'].dt.year

# reproject to BC Albers to match fire perimeters
hotspots = hotspots.to_crs('EPSG:3005')

print(f"\nHotspots loaded: {len(hotspots)}")
print(f"Years: {sorted(hotspots['year'].unique())}")
print(f"CRS: {hotspots.crs}")

# =============================================================================
# STEP 2 — SPLIT HOTSPOTS BY YEAR AND SEASON
# =============================================================================

def split_hotspots_fall_spring(hotspots_year, year, sdd_lookup,
                                fall_doy_start=FALL_DOY_START):
    """
    Split annual hotspots into fall and spring subsets.

    Fall:   DOY >= fall_doy_start (Aug 1) — late season before winter dormancy
    Spring: DOY >= min SDD across zones — early season after snowmelt
            Upper bound (SDD + 50 days) applied per-fire in detect_overwintering

    Parameters
    ----------
    hotspots_year : GeoDataFrame
        Hotspots for a single year
    year : int
        Year of hotspots
    sdd_lookup : dict
        SDD DOY per fire zone and year
    fall_doy_start : int
        DOY cutoff for fall subset (default 213 = Aug 1)

    Returns
    -------
    fall : GeoDataFrame
    spring : GeoDataFrame
    """
    fall = hotspots_year[hotspots_year['doy'] >= fall_doy_start].copy()

    sdd_values = [
        sdd_lookup[zone][year]
        for zone in sdd_lookup
        if year in sdd_lookup[zone]
    ]

    if sdd_values:
        min_sdd = min(sdd_values)
        spring  = hotspots_year[hotspots_year['doy'] >= min_sdd].copy()
    else:
        print(f"  No SDD found for year {year}, spring subset will be empty")
        spring  = gpd.GeoDataFrame(
            columns=hotspots_year.columns, crs=hotspots_year.crs
        )
        min_sdd = 'N/A'

    print(f"Year {year} | fall: {len(fall)} hotspots (DOY >= {fall_doy_start}) "
          f"| spring: {len(spring)} hotspots (DOY >= {min_sdd})")

    return fall, spring


# split by year
hotspots_2023 = hotspots[hotspots['year'] == 2023].copy()
hotspots_2024 = hotspots[hotspots['year'] == 2024].copy()
hotspots_2025 = hotspots[hotspots['year'] == 2025].copy()

# split by season:
# 2023 — fall only (no spring, 2023 is the initial fire year)
# 2024 — fall (for 2024->2025) and spring (for 2023->2024)
# 2025 — spring only (no fall needed)
fall_2023, _           = split_hotspots_fall_spring(hotspots_2023, 2023, SDD_LOOKUP)
fall_2024, spring_2024 = split_hotspots_fall_spring(hotspots_2024, 2024, SDD_LOOKUP)
_,         spring_2025 = split_hotspots_fall_spring(hotspots_2025, 2025, SDD_LOOKUP)

# =============================================================================
# STEP 3 — DETECT OVERWINTERING FIRES
# =============================================================================
# For each fire perimeter:
#   1. Find fall hotspots within perimeter (>= 2 detections for persistence)
#   2. Find spring hotspots within buffer between SDD and SDD + 50 days
#   3. Find closest spring hotspot to last fall hotspot location
#   4. Classify as overwintering if travel distance <= 1000m (Scholten et al.)

def detect_overwintering(fires, buffered_dfs, fall_hotspots, spring_hotspots,
                         buffer_dist, fire_year, spring_year):
    """
    Detect overwintering fires using Scholten et al. criteria.

    Parameters
    ----------
    fires : GeoDataFrame
        Fire perimeters with fire_id, fire_year, fire_zone columns
    buffered_dfs : dict
        Buffered perimeters keyed by buffer distance, with SDD columns
    fall_hotspots : GeoDataFrame
        Hotspots from fall of fire_year
    spring_hotspots : GeoDataFrame
        Hotspots from spring of spring_year
    buffer_dist : int
        Buffer distance in metres
    fire_year : int
        Year fire started (e.g. 2023)
    spring_year : int
        Year of spring reactivation (e.g. 2024)

    Returns
    -------
    list of dicts, one record per fire
    """
    perims  = fires[fires['fire_year'] == float(fire_year)].copy()
    buffers = buffered_dfs[buffer_dist][
        buffered_dfs[buffer_dist]['fire_year'] == float(fire_year)
    ].copy()

    results = []

    for _, fire in perims.iterrows():
        fid       = fire['fire_id']
        fire_zone = fire['fire_zone']
        fire_geom = fire['geometry']

        # get buffer geometry for this fire
        buf_row = buffers[buffers['fire_id'] == fid]
        if buf_row.empty:
            continue
        buf_row  = buf_row.iloc[0]
        buf_geom = buf_row['geometry']

        # get per-fire per-buffer SDD (mean DOY)
        sdd_col = f'sdd_{spring_year}_mean'
        if sdd_col not in buf_row.index or pd.isna(buf_row[sdd_col]):
            continue
        sdd_doy = buf_row[sdd_col]

        # --- fall hotspots within perimeter ---
        fall_in_perim = fall_hotspots[fall_hotspots.within(fire_geom)]

        # --- spring hotspots within buffer between SDD and SDD + 50 days ---
        spring_in_window = spring_hotspots[
            (spring_hotspots['doy'] >= sdd_doy) &
            (spring_hotspots['doy'] <= sdd_doy + SDD_MAX_DAYS)
        ]
        spring_in_buffer = spring_in_window[spring_in_window.within(buf_geom)]

        # require >= 2 fall detections (persistence check)
        has_fall   = len(fall_in_perim) >= 1
        has_spring = len(spring_in_buffer) > 0

        # --- last fall hotspot ---
        if has_fall:
            last_fall_idx  = fall_in_perim['datetime'].idxmax()
            last_fall      = fall_in_perim.loc[last_fall_idx]
            last_fall_date = last_fall['datetime']
            last_fall_lat  = last_fall['latitude']
            last_fall_lon  = last_fall['longitude']
            last_fall_geom = last_fall['geometry']
        else:
            last_fall_date = last_fall_lat = last_fall_lon = last_fall_geom = None

        # --- spring hotspot analysis ---
        if has_spring and last_fall_geom is not None:
            spring_in_buffer = spring_in_buffer.copy()

            # first spring hotspot by date
            first_spring_by_date = spring_in_buffer.loc[
                spring_in_buffer['datetime'].idxmin()
            ]

            # closest spring hotspot to last fall hotspot location
            # physically meaningful: overwintering fire should reactivate
            # near where it was last smoldering
            spring_in_buffer['dist_to_last_fall'] = (
                spring_in_buffer['geometry'].distance(last_fall_geom)
            )
            closest_spring = spring_in_buffer.loc[
                spring_in_buffer['dist_to_last_fall'].idxmin()
            ]

            first_spring_date         = first_spring_by_date['datetime']
            first_spring_lat          = first_spring_by_date['latitude']
            first_spring_lon          = first_spring_by_date['longitude']
            first_spring_closest_date = closest_spring['datetime']
            first_spring_closest_lat  = closest_spring['latitude']
            first_spring_closest_lon  = closest_spring['longitude']
            first_spring_closest_dist_to_fall  = closest_spring['dist_to_last_fall']
            first_spring_closest_dist_to_perim = closest_spring['geometry'].distance(
                fire_geom.boundary
            )

            # travel distance in projected metres (BC Albers)
            lf_x, lf_y = TRANSFORMER.transform(last_fall_lon, last_fall_lat)
            cs_x, cs_y = TRANSFORMER.transform(
                first_spring_closest_lon, first_spring_closest_lat
            )
            travel_dist = euclidean([lf_x, lf_y], [cs_x, cs_y])

        else:
            first_spring_date                  = None
            first_spring_lat                   = None
            first_spring_lon                   = None
            first_spring_closest_date          = None
            first_spring_closest_lat           = None
            first_spring_closest_lon           = None
            first_spring_closest_dist_to_fall  = None
            first_spring_closest_dist_to_perim = None
            travel_dist                        = None

        # --- overwintering classification (Scholten et al.) ---
        overwintered = (
            has_fall and
            has_spring and
            travel_dist is not None and
            travel_dist <= MAX_HOTSPOT_DISTANCE_M
        )

        results.append({
            'fire_id':                             fid,
            'fire_year':                           fire_year,
            'fire_zone':                           fire_zone,
            'buffer_dist':                         buffer_dist,
            'spring_year':                         spring_year,
            'sdd_doy':                             sdd_doy,
            'sdd_max_doy':                         sdd_doy + SDD_MAX_DAYS,
            'n_fall_hotspots':                     len(fall_in_perim),
            'n_spring_hotspots':                   len(spring_in_buffer),
            'has_fall_hotspots':                   has_fall,
            'has_spring_hotspots':                 has_spring,
            'travel_dist_m':                       travel_dist,
            'within_distance_threshold':           (
                travel_dist <= MAX_HOTSPOT_DISTANCE_M
                if travel_dist is not None else False
            ),
            'overwintered':                        overwintered,
            # last fall hotspot
            'last_fall_date':                      last_fall_date,
            'last_fall_lat':                       last_fall_lat,
            'last_fall_lon':                       last_fall_lon,
            # first spring hotspot by date
            'first_spring_date':                   first_spring_date,
            'first_spring_lat':                    first_spring_lat,
            'first_spring_lon':                    first_spring_lon,
            # closest spring hotspot to last fall hotspot
            'first_spring_closest_date':           first_spring_closest_date,
            'first_spring_closest_lat':            first_spring_closest_lat,
            'first_spring_closest_lon':            first_spring_closest_lon,
            'first_spring_closest_dist_to_fall':   first_spring_closest_dist_to_fall,
            'first_spring_closest_dist_to_perim':  first_spring_closest_dist_to_perim,
            # TODO: add when data available
            'flag_near_infrastructure':            None,
            'flag_near_lightning':                 None,
        })

    return results


# run detection for all buffer distances and year combinations
all_results = {}
for dist in BUFFER_DISTANCES:
    results_23_24 = detect_overwintering(
        fires, buffered_dfs, fall_2023, spring_2024,
        dist, fire_year=2023, spring_year=2024
    )
    results_24_25 = detect_overwintering(
        fires, buffered_dfs, fall_2024, spring_2025,
        dist, fire_year=2024, spring_year=2025
    )

    # checkpoint before converting to dataframe
    print(f"\nBuffer {dist}m:")
    print(f"  results_23_24 length: {len(results_23_24)}")
    print(f"  results_24_25 length: {len(results_24_25)}")

    if len(results_23_24) == 0 or len(results_24_25) == 0:
        # diagnose why function returned empty
        perims_2023 = fires[fires['fire_year'] == 2023.0]
        bufs_2023   = buffered_dfs[dist][buffered_dfs[dist]['fire_year'] == 2023.0]
        sdd_col     = f'sdd_2024_mean'
        print(f"  2023 perimeters: {len(perims_2023)}")
        print(f"  2023 buffers at {dist}m: {len(bufs_2023)}")
        print(f"  SDD col present: {sdd_col in buffered_dfs[dist].columns}")
        print(f"  fall_2023 rows: {len(fall_2023)}")
        print(f"  spring_2024 rows: {len(spring_2024)}")
        print(f"  fall_2023 CRS: {fall_2023.crs}")
        print(f"  fires CRS: {fires.crs}")
        print(f"  buffered_dfs CRS: {buffered_dfs[dist].crs}")
        break

    df_23_24 = pd.DataFrame(results_23_24)
    df_24_25 = pd.DataFrame(results_24_25)

    all_results[dist] = pd.concat([df_23_24, df_24_25], ignore_index=True)

    print(f"  2023->2024 overwintering: "
          f"{df_23_24['overwintered'].sum()} of {len(df_23_24)} fires")
    print(f"  2024->2025 overwintering: "
          f"{df_24_25['overwintered'].sum()} of {len(df_24_25)} fires")

# =============================================================================
# STEP 4 — CALCULATE OVERWINTERING METRICS
# =============================================================================
# For each overwintering fire calculate:
#   - hotspot_distance_m: euclidean distance between last fall and closest spring hotspot
#   - dormancy_days: days between last fall and closest spring hotspot
#   - days_after_sdd: days between closest spring hotspot and SDD
#   - dist_to_prev_perimeter_m: distance of closest spring hotspot to perimeter edge

def calculate_overwintering_metrics(all_results, fires, buffer_distances):
    """
    Calculate distance and time metrics for overwintering fires.
    All calculations use the closest spring hotspot to last fall hotspot,
    which is more physically meaningful than the earliest-by-date hotspot
    for large fire perimeters.
    """
    for dist in buffer_distances:
        df   = all_results[dist].copy()
        mask = df['overwintered'] == True

        df['last_fall_date']            = pd.to_datetime(df['last_fall_date'])
        df['first_spring_closest_date'] = pd.to_datetime(df['first_spring_closest_date'])

        if mask.sum() == 0:
            print(f"Buffer {dist}m: no overwintering fires found")
            all_results[dist] = df
            continue

        # project to BC Albers for accurate distance in metres
        lf_x, lf_y = TRANSFORMER.transform(
            df.loc[mask, 'last_fall_lon'].values,
            df.loc[mask, 'last_fall_lat'].values
        )
        cs_x, cs_y = TRANSFORMER.transform(
            df.loc[mask, 'first_spring_closest_lon'].values,
            df.loc[mask, 'first_spring_closest_lat'].values
        )

        # euclidean distance between last fall and closest spring hotspot
        df.loc[mask, 'hotspot_distance_m'] = [
            euclidean([x1, y1], [x2, y2])
            for x1, y1, x2, y2 in zip(lf_x, lf_y, cs_x, cs_y)
        ]

        # dormancy duration in days
        df.loc[mask, 'dormancy_days'] = (
            df.loc[mask, 'first_spring_closest_date'] -
            df.loc[mask, 'last_fall_date']
        ).dt.days

        # days after SDD of closest spring hotspot
        df.loc[mask, 'first_spring_doy'] = pd.to_datetime(
            df.loc[mask, 'first_spring_closest_date']
        ).dt.dayofyear
        df.loc[mask, 'days_after_sdd'] = (
            df.loc[mask, 'first_spring_doy'] - df.loc[mask, 'sdd_doy']
        )

        # distance from closest spring hotspot to previous year perimeter edge
        dist_to_perim = []
        for i, (idx, row) in enumerate(df[mask].iterrows()):
            perim = fires[fires['fire_id'] == row['fire_id']]
            if perim.empty:
                dist_to_perim.append(None)
                continue
            spring_point = Point(cs_x[i], cs_y[i])
            dist_to_perim.append(
                spring_point.distance(perim.iloc[0]['geometry'].boundary)
            )

        df.loc[mask, 'dist_to_prev_perimeter_m'] = dist_to_perim
        all_results[dist] = df

        print(f"\nBuffer {dist}m — overwintering metrics:")
        cols = ['fire_id', 'fire_year', 'spring_year', 'hotspot_distance_m',
                'dormancy_days', 'days_after_sdd', 'dist_to_prev_perimeter_m']
        print(df[mask][cols].to_string())

    return all_results


all_results = calculate_overwintering_metrics(all_results, fires, BUFFER_DISTANCES)

# =============================================================================
# STEP 5 — IDENTIFY MULTI-YEAR OVERWINTERING FIRES (2023 -> 2024 -> 2025)
# =============================================================================
# A confirmed multi-year chain requires:
#   1. 2023 fire overwintered into 2024 (spring 2024 hotspot in 2023 buffer)
#   2. Spring 2024 hotspot lands within/near a 2024 perimeter (event linkage)
#   3. That 2024 fire overwintered into 2025 (spring 2025 hotspot in 2024 buffer)
#   4. Spring 2025 hotspot lands within/near a 2025 perimeter (event linkage)
# Perimeter linkage uses distance-based matching (<=500m) to account for
# VIIRS positional uncertainty (~375m nominal pixel size)

def match_hotspot_to_nearest_perimeter(hotspot_gdf, perims, id_col_left,
                                        id_col_right, max_dist_m=500):
    """
    Match each hotspot point to the nearest perimeter within max_dist_m.
    Distance-based matching accounts for VIIRS positional uncertainty.

    Parameters
    ----------
    hotspot_gdf : GeoDataFrame
        Hotspot points with id_col_left column and geometry
    perims : GeoDataFrame
        Fire perimeters with id_col_right column and geometry
    id_col_left : str
        ID column in hotspot_gdf
    id_col_right : str
        ID column in perims
    max_dist_m : float
        Maximum matching distance in metres

    Returns
    -------
    pd.DataFrame with id_col_left, id_col_right, dist_to_perimeter
    """
    records = []
    for _, row in hotspot_gdf.iterrows():
        pt          = row['geometry']
        perims_copy = perims.copy()
        perims_copy['dist'] = perims_copy['geometry'].distance(pt)
        nearest = perims_copy[perims_copy['dist'] <= max_dist_m]
        for _, perim_row in nearest.iterrows():
            records.append({
                id_col_left:         row[id_col_left],
                id_col_right:        perim_row[id_col_right],
                'dist_to_perimeter': perim_row['dist']
            })
    return pd.DataFrame(records)


def find_multiyear_overwintering(all_results, fires, buffer_distances,
                                  hotspot_buffer_m=500):
    """
    Identify confirmed multi-year overwintering fire chains (2023->2024->2025).

    Parameters
    ----------
    all_results : dict
        Overwintering results keyed by buffer distance
    fires : GeoDataFrame
        Fire perimeters
    buffer_distances : list
        Buffer distances to analyse
    hotspot_buffer_m : float
        Max distance for hotspot-to-perimeter matching (default 500m)

    Returns
    -------
    dict of DataFrames keyed by buffer distance
    """
    multiyear_results = {}

    for dist in buffer_distances:
        df = all_results[dist].copy()

        # --- 2023 fires that overwintered into 2024 ---
        ow_23_24 = df[
            (df['fire_year'] == 2023) &
            (df['overwintered'] == True) &
            (df['spring_year'] == 2024)
        ][['fire_id', 'fire_zone', 'n_fall_hotspots', 'n_spring_hotspots',
           'hotspot_distance_m', 'dormancy_days', 'days_after_sdd',
           'dist_to_prev_perimeter_m', 'last_fall_date', 'first_spring_date',
           'first_spring_closest_lat', 'first_spring_closest_lon']].copy()

        ow_23_24 = ow_23_24.rename(columns={
            'fire_id':                  'fire_id_2023',
            'n_fall_hotspots':          'n_fall_hotspots_2023',
            'n_spring_hotspots':        'n_spring_hotspots_2024',
            'hotspot_distance_m':       'hotspot_distance_m_2324',
            'dormancy_days':            'dormancy_days_2324',
            'days_after_sdd':           'days_after_sdd_2324',
            'dist_to_prev_perimeter_m': 'dist_to_perimeter_2324',
            'last_fall_date':           'last_fall_date_2023',
            'first_spring_date':        'first_spring_date_2024'
        })

        # --- 2024 fires that overwintered into 2025 ---
        ow_24_25 = df[
            (df['fire_year'] == 2024) &
            (df['overwintered'] == True) &
            (df['spring_year'] == 2025)
        ][['fire_id', 'n_fall_hotspots', 'n_spring_hotspots',
           'hotspot_distance_m', 'dormancy_days', 'days_after_sdd',
           'dist_to_prev_perimeter_m', 'last_fall_date', 'first_spring_date',
           'first_spring_closest_lat', 'first_spring_closest_lon']].copy()

        ow_24_25 = ow_24_25.rename(columns={
            'fire_id':                  'fire_id_2024',
            'n_fall_hotspots':          'n_fall_hotspots_2024',
            'n_spring_hotspots':        'n_spring_hotspots_2025',
            'hotspot_distance_m':       'hotspot_distance_m_2425',
            'dormancy_days':            'dormancy_days_2425',
            'days_after_sdd':           'days_after_sdd_2425',
            'dist_to_prev_perimeter_m': 'dist_to_perimeter_2425',
            'last_fall_date':           'last_fall_date_2024',
            'first_spring_date':        'first_spring_date_2025'
        })

        print(f"\nBuffer {dist}m:")
        print(f"  2023 fires overwintered into 2024: {len(ow_23_24)}")
        print(f"  2024 fires overwintered into 2025: {len(ow_24_25)}")

        if ow_23_24.empty or ow_24_25.empty:
            print(f"  No multi-year candidates possible")
            multiyear_results[dist] = pd.DataFrame()
            continue

        # --- link spring 2024 hotspot to 2024 perimeter ---
        perims_2024 = fires[fires['fire_year'] == 2024.0][['fire_id', 'geometry']].rename(
            columns={'fire_id': 'fire_id_2024'}
        )
        ow_23_24_gdf = gpd.GeoDataFrame(
            ow_23_24,
            geometry=gpd.points_from_xy(
                ow_23_24['first_spring_closest_lon'],
                ow_23_24['first_spring_closest_lat']
            ),
            crs='EPSG:4326'
        ).to_crs('EPSG:3005')

        hotspot_in_2024_perim = match_hotspot_to_nearest_perimeter(
            ow_23_24_gdf[['fire_id_2023', 'geometry']],
            perims_2024,
            id_col_left='fire_id_2023',
            id_col_right='fire_id_2024',
            max_dist_m=hotspot_buffer_m
        )

        print(f"  Spring 2024 hotspots matched to 2024 perimeters: "
              f"{len(hotspot_in_2024_perim)}")

        if hotspot_in_2024_perim.empty:
            print(f"  No 2024 perimeter matches found")
            multiyear_results[dist] = pd.DataFrame()
            continue

        # --- link spring 2025 hotspot to 2025 perimeter ---
        perims_2025 = fires[fires['fire_year'] == 2025.0][['fire_id', 'geometry']].rename(
            columns={'fire_id': 'fire_id_2025'}
        )
        ow_24_25_gdf = gpd.GeoDataFrame(
            ow_24_25,
            geometry=gpd.points_from_xy(
                ow_24_25['first_spring_closest_lon'],
                ow_24_25['first_spring_closest_lat']
            ),
            crs='EPSG:4326'
        ).to_crs('EPSG:3005')

        hotspot_in_2025_perim = match_hotspot_to_nearest_perimeter(
            ow_24_25_gdf[['fire_id_2024', 'geometry']],
            perims_2025,
            id_col_left='fire_id_2024',
            id_col_right='fire_id_2025',
            max_dist_m=hotspot_buffer_m
        )

        # aggregate 2025 fire ids into list per 2024 fire
        hotspot_in_2025_perim_agg = (
            hotspot_in_2025_perim
            .groupby('fire_id_2024')['fire_id_2025']
            .apply(list)
            .reset_index()
        )

        print(f"  Spring 2025 hotspots matched to 2025 perimeters: "
              f"{len(hotspot_in_2025_perim_agg)}")

        if hotspot_in_2025_perim_agg.empty:
            print(f"  No 2025 perimeter matches found")
            multiyear_results[dist] = pd.DataFrame()
            continue

        # --- build full 2023->2024->2025 chain ---
        multiyear = (
            hotspot_in_2024_perim[['fire_id_2023', 'fire_id_2024', 'dist_to_perimeter']]
            .merge(hotspot_in_2025_perim_agg, on='fire_id_2024', how='inner')
            .merge(
                ow_23_24.drop(columns=[
                    'first_spring_closest_lat', 'first_spring_closest_lon'
                ]),
                on='fire_id_2023', how='inner'
            )
            .merge(
                ow_24_25.drop(columns=[
                    'first_spring_closest_lat', 'first_spring_closest_lon'
                ]),
                on='fire_id_2024', how='inner'
            )
        )

        multiyear['buffer_dist'] = dist
        multiyear_results[dist]  = multiyear

        print(f"  Confirmed multi-year overwintering fires: {len(multiyear)}")
        print(multiyear[[
            'fire_id_2023', 'fire_id_2024', 'fire_id_2025', 'fire_zone',
            'dormancy_days_2324', 'dormancy_days_2425',
            'days_after_sdd_2324', 'days_after_sdd_2425'
        ]].to_string())

    return multiyear_results


multiyear_results = find_multiyear_overwintering(
    all_results, fires, BUFFER_DISTANCES
)

# =============================================================================
# STEP 6 — SUMMARY STATISTICS PER FIRE ZONE AND YEAR
# =============================================================================

def calculate_summary_statistics(all_results, multiyear_results, buffer_distances):
    """
    Summary statistics per fire zone and spring year.

    Metrics:
        n_total                : total fires assessed
        n_overwintering        : fires classified as overwintering
        n_non_overwintering    : fires not classified as overwintering
        n_multiyear            : confirmed multi-year chains (2023->2024->2025)
        avg_hotspot_distance_m : mean travel distance (fall to spring hotspot)
        avg_dormancy_days      : mean dormancy duration in days
        avg_days_after_sdd     : mean days after snowmelt of spring reactivation
        avg_dist_to_perimeter  : mean distance of spring hotspot to perimeter edge
    """
    for dist in buffer_distances:
        df = all_results[dist].copy()

        summary = df.groupby(['fire_zone', 'spring_year']).agg(
            n_total               = ('fire_id', 'count'),
            n_overwintering       = ('overwintered', 'sum'),
            n_non_overwintering   = ('overwintered', lambda x: (~x).sum()),
            avg_hotspot_distance_m= ('hotspot_distance_m', 'mean'),
            avg_dormancy_days     = ('dormancy_days', 'mean'),
            avg_days_after_sdd    = ('days_after_sdd', 'mean'),
            avg_dist_to_perimeter = ('dist_to_prev_perimeter_m', 'mean')
        ).reset_index()

        # multi-year counts — fire_zone already present in multiyear_results
        if dist in multiyear_results and not multiyear_results[dist].empty:
            multiyear_counts = (
                multiyear_results[dist]
                .groupby('fire_zone')
                .size()
                .reset_index(name='n_multiyear')
            )
            summary = summary.merge(multiyear_counts, on='fire_zone', how='left')
            summary['n_multiyear'] = summary['n_multiyear'].fillna(0).astype(int)
        else:
            summary['n_multiyear'] = 0

        print(f"\nBuffer {dist}m — summary statistics:")
        print(summary.to_string())

        summary.to_csv(
            f"data/analysis/summary_statistics_{dist}m.csv", index=False
        )
        print(f"Saved summary_statistics_{dist}m.csv")


calculate_summary_statistics(all_results, multiyear_results, BUFFER_DISTANCES)

# =============================================================================
# STEP 7 — SAVE ALL RESULTS
# =============================================================================

for dist in BUFFER_DISTANCES:
    # full overwintering classification and metrics for all fires
    all_results[dist].to_csv(
        f"data/analysis/overwintering_results_{dist}m.csv", index=False
    )

    # confirmed multi-year overwintering chains
    if dist in multiyear_results and not multiyear_results[dist].empty:
        multiyear_results[dist].to_csv(
            f"data/analysis/multiyear_overwintering_{dist}m.csv", index=False
        )

    print(f"Saved results for buffer {dist}m")

print("\nAll results saved.")