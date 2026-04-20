# script to find hotspots that fall within the buffer of each fire perimeter from the preceeding year
# hotspot files:
# - ../data/processed_hotspots/hotspots_clipped_2023.geojson
# - ../data/processed_hotspots/hotspots_clipped_2024.geojson
# - ../data/processed_hotspots/hotspots_clipped_2025.geojson
#
#
# buffer files:
# - "../data/processed_wildfire_buffers/perims_2023_clipped_buffers.parquet"
# - "../data/processed_wildfire_buffers/perims_2024_clipped_buffers.parquet"
# - "../data/processed_wildfire_buffers/perims_2025_clipped_buffers.parquet"
#
# For the buffer files, the parquet contains two geometry columns: `geometry` which is the original fire perimeter geometry, and `buffer_wkt` which is the buffered geometry. We want to use `buffer_wkt` for the spatial join to find hotspots that fall within the buffer of each fire perimeter.
# buffer_wkt is a WKT string, so we will need to convert it to a geometry object before doing the spatial join. We can use shapely.wkt.loads to convert the WKT string to a geometry object.
# crs is store in CRS column, and it is in the format "EPSG:XXXX", so we can use that to set the CRS of the GeoDataFrame before doing the spatial join.
#
# We want a function that takes a hotspot file for a given year, and a buffer file for the preceeding year
# and returns a GeoDataFrame of hotspots that fall within the buffer of each fire perimeter from the preceeding year. We can then save this GeoDataFrame as a new file for each year.
# For example, for 2024 hotspots, we would use the 2023 buffer file to find hotspots that fall within the buffer of each fire perimeter from 2023, and save the result as "hotspots_in_buffers_2024.shp" and "hotspots_in_buffers_2024.geojson". We would repeat this process for 2025 hotspots using the 2024 buffer file.
# function will be run from within a .ipynb file, so we can import it and call it for each year.
# When hotspot are found within the buffer of a fire perimeter, we want to add a column to the resulting GeoDataFrame that indicates which fire perimeter it was found within. We can use the index of the fire perimeter in the buffer file as the identifier for the fire perimeter, and add a column called "fire_perimeter_id" to the resulting GeoDataFrame that contains this identifier for each hotspot that falls within a buffer.
# We also want to add a column that indicates the year of the hotspots, which we can extract from the ACQ_DATE column in the hotspot file. We can call this column "hotspot_year".
# We also want to calculate the distance (euclidean) from each hotspot to the original fire perimeter contains in the 'geometry' column which is also a wkt string, and add this as a new column called "distance_to_perimeter". We can use shapely to calculate the distance between the hotspot geometry and the original fire perimeter geometry.
# We also want a function that calculates the last time a hotspot was observed within a fire perimter, so for year 2023 fire perimters, when was the last time a hotspot was found within the 'geometry'. Using this hotspot, what is the distance between the last hotspot found, and the hotspot from 2024 identified within the buffer_wkt. This will give us a sense of how far the fire has spread from the last known hotspot to the new hotspot found within the buffer. We can add this as a new column called "distance_from_last_hotspot". To calculate this, we can first find the last hotspot observed within each fire perimeter in 2023, and then calculate the distance from this last hotspot to each new hotspot found within the buffer in 2024. We can use shapely to calculate this distance as well.
# Finally, we want to save the resulting GeoDataFrame for each year as a new file, with the name "hotspots_in_buffers_YEAR.shp" and "hotspots_in_buffers_YEAR.geojson", where YEAR is the year of the hotspots (2024 or 2025).

import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely import wkt

ROOT = Path(__file__).parent.parent
HOTSPOT_DIR = ROOT / "data" / "processed_hotspots"
BUFFER_DIR = ROOT / "data" / "processed_wildfire_buffers"
OUT_DIR = ROOT / "data" / "processed_hotspots"


def _load_buffers(buffer_year: int) -> gpd.GeoDataFrame:
    """Load a buffer parquet, convert WKT columns to geometry, return two GDFs."""
    path = BUFFER_DIR / f"perims_{buffer_year}_clipped_buffers.parquet"
    df = pd.read_parquet(path)

    crs = df["crs"].iloc[0]

    # Original perimeter geometry
    perims = gpd.GeoDataFrame(
        df.drop(columns=["buffer_wkt"]),
        geometry=df["geometry"].apply(wkt.loads),
        crs=crs,
    )
    perims = perims.reset_index(drop=True)
    perims["fire_perimeter_id"] = perims.index

    # Buffer geometry (for spatial join)
    buffers = gpd.GeoDataFrame(
        df[["fire_perimeter_id"] if "fire_perimeter_id" in df.columns else []].assign(
            fire_perimeter_id=perims.index
        ),
        geometry=df["buffer_wkt"].apply(wkt.loads),
        crs=crs,
    )

    return perims, buffers


def _load_hotspots(year: int) -> gpd.GeoDataFrame:
    path = HOTSPOT_DIR / f"hotspots_clipped_{year}.geojson"
    gdf = gpd.read_file(path)
    gdf["ACQ_DATE"] = pd.to_datetime(gdf["ACQ_DATE"])
    return gdf


def _last_hotspot_per_perimeter(
    perims: gpd.GeoDataFrame, prev_year_hotspots: gpd.GeoDataFrame
) -> dict:
    """
    For each fire perimeter, find the last (latest ACQ_DATE) hotspot from the
    preceding year that fell within the original perimeter boundary.

    Returns a dict mapping fire_perimeter_id ->
        {"geometry": Point | None, "date": Timestamp | None}
    """
    spots = prev_year_hotspots.to_crs(perims.crs)

    result: dict = {}
    for _, perim_row in perims.iterrows():
        pid = perim_row["fire_perimeter_id"]
        inside = spots[spots.within(perim_row.geometry)]
        if inside.empty:
            result[pid] = {"geometry": None, "date": None}
        else:
            latest = inside.loc[inside["ACQ_DATE"].idxmax()]
            result[pid] = {"geometry": latest.geometry, "date": latest["ACQ_DATE"]}
    return result


def find_hotspots_in_buffers(hotspot_year: int, save: bool = True) -> gpd.GeoDataFrame:
    """
    Find hotspots from `hotspot_year` that fall within the buffered perimeters
    of fires from the preceding year.

    Parameters
    ----------
    hotspot_year : int
        Year of hotspot data to use (e.g. 2024 or 2025).
    save : bool
        If True, write results to data/processed_hotspots/.

    Returns
    -------
    GeoDataFrame with one row per hotspot-in-buffer, augmented with:
      - fire_perimeter_id     : index of the matched preceding-year perimeter
      - hotspot_year          : year extracted from ACQ_DATE
      - distance_to_perimeter : Euclidean distance (m) from hotspot to original perimeter edge
      - distance_from_last_hotspot : distance (m) from the last hotspot observed
                                     inside the perimeter in the preceding year
    """
    buffer_year = hotspot_year - 1

    print(f"Loading {hotspot_year} hotspots and {buffer_year} buffers...")
    hotspots = _load_hotspots(hotspot_year)
    perims, buffers = _load_buffers(buffer_year)

    # Ensure CRS alignment
    hotspots = hotspots.to_crs(buffers.crs)

    # Spatial join: hotspots within buffer polygons
    print("Running spatial join (hotspots within buffers)...")
    joined = gpd.sjoin(
        hotspots,
        buffers[["fire_perimeter_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"])
    joined["hotspot_year"] = joined["ACQ_DATE"].dt.year

    # Attach original perimeter geometry for distance calculations
    perim_geom_lookup = perims.set_index("fire_perimeter_id")["geometry"].to_dict()

    print("Calculating distance_to_perimeter...")
    joined["distance_to_perimeter"] = joined.apply(
        lambda row: row.geometry.distance(perim_geom_lookup[row["fire_perimeter_id"]]),
        axis=1,
    )

    # Find last hotspot per perimeter in the preceding year
    print(f"Finding last hotspot per perimeter in {buffer_year}...")
    prev_hotspots = _load_hotspots(buffer_year)
    last_geom_by_perim = _last_hotspot_per_perimeter(perims, prev_hotspots)

    # Add last-detection date from preceding year (within original perimeter)
    joined["last_hotspot_date_in_perimeter"] = joined["fire_perimeter_id"].map(
        {pid: v["date"] for pid, v in last_geom_by_perim.items()}
    )

    print("Calculating distance_from_last_hotspot...")
    def _dist_from_last(row):
        last = last_geom_by_perim.get(row["fire_perimeter_id"], {}).get("geometry")
        if last is None:
            return float("nan")
        return row.geometry.distance(last)

    joined["distance_from_last_hotspot"] = joined.apply(_dist_from_last, axis=1)

    print(f"  {len(joined)} hotspots found in {buffer_year} buffers")

    if save:
        shp_path = OUT_DIR / f"hotspots_in_buffers_{hotspot_year}.shp"
        gjson_path = OUT_DIR / f"hotspots_in_buffers_{hotspot_year}.geojson"
        joined.to_file(shp_path)
        joined.to_file(gjson_path, driver="GeoJSON")
        print(f"  saved {shp_path.name} and {gjson_path.name}")

    return joined
