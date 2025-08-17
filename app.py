import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import tempfile
import zipfile
import os
from io import BytesIO
import ee
import geemap
import math
import numpy as np

from google.oauth2 import service_account

def init_ee():
    creds_dict = dict(st.secrets["GEE"])

    scopes = [
        "https://www.googleapis.com/auth/earthengine.readonly",
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/devstorage.full_control"
    ]

    google_creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=scopes
    )

    ee.Initialize(google_creds, project=creds_dict["project_id"])

# Initialize Earth Engine via service account
init_ee()

# ---------------- Helpers ----------------
def transform_coord(value):
    step1 = value / 100.0
    step2 = int(step1)
    step3 = step1 - step2
    step4 = step3 * 100
    step5 = step4 * 100 / 60
    step6 = step5 / 100
    step7 = step6 + step2
    return step7

def mask_clouds_qa60(image):
    # Classic QA60 mask: remove cloud + cirrus
    qa = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

def build_composite(bounds, start_date, end_date):
    # Returns 4-band (B4,B3,B2,B8) cloud-masked median as int16 (0..10000)
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bounds)
        .filterDate(str(start_date), str(end_date))
        .map(mask_clouds_qa60)
    )
    return col.select(["B4", "B3", "B2", "B8"]).median().toInt16()

def estimate_pixels(lat1, lat2, lon1, lon2, scale_m=10):
    # Very rough pixel estimate (good enough to decide tiling)
    phi = (lat1 + lat2) / 2.0
    h_m = abs(lat1 - lat2) * 110_540.0
    w_m = abs(lon1 - lon2) * 111_320.0 * max(0.1, math.cos(math.radians(phi)))
    return (h_m / scale_m) * (w_m / scale_m)

def safe_tile_steps(lat1, lat2, lon1, lon2, scale_m=10, bands=4, bytes_per_sample=2, limit_mb=48):
    """
    Pick lat/lon step so each tile stays well under the ~50 MB synchronous download cap.
    Assumes int16 outputs (2 bytes per sample).
    """
    limit_bytes = limit_mb * 1024 * 1024
    safe_side_m = scale_m * math.sqrt(limit_bytes / float(bands * bytes_per_sample))
    safe_side_m *= 0.70  # margin (~30%)
    # meters -> degrees
    phi = (lat1 + lat2) / 2.0
    dlat = safe_side_m / 110_540.0
    dlon = safe_side_m / (111_320.0 * max(0.1, math.cos(math.radians(phi))))
    # never let a tile get too big (also avoids EE reprojection issues)
    return min(dlat, 0.35), min(dlon, 0.35)

def arange_edges(start, stop, step):
    vals = []
    v = start
    while v < stop - 1e-12:
        vals.append(v)
        v += step
    vals.append(stop)
    return vals

def clear_fraction_qa60(bounds, start_date, end_date, sample_scale=500):
    """
    Fraction of AOI pixels that have at least one clear (QA60 cloud+cirrus-free) observation
    within the date range. Computed at coarse resolution for speed.
    """
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bounds)
        .filterDate(str(start_date), str(end_date))
    )

    def clear01(img):
        qa = img.select("QA60")
        is_clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return is_clear.rename("clear").unmask(0)

    clears = col.map(clear01)
    any_clear = clears.max()  # per-pixel: 1 if clear in any image, else 0
    stats = any_clear.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=bounds,
        scale=sample_scale,      # coarse sampling for speed
        maxPixels=1e8
    )
    return ee.Number(stats.get("clear"))

# ---------------- UI ----------------
st.title("Excel to shapefile converter")
st.write("Upload your Excel file, choose whether data is raw or already transformed, and download a shapefile.")

# Initialize session state variables
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "composite_image_bytes" not in st.session_state:
    st.session_state["composite_image_bytes"] = None
if "tiles_zip_bytes" not in st.session_state:
    st.session_state["tiles_zip_bytes"] = None

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

# Add choice for transformation
transform_choice = st.radio(
    "Is your uploaded sheet already transformed?",
    ["Already transformed", "Raw sheet (needs transformation)"]
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_name:
        st.session_state.zip_bytes = None
        st.session_state.last_uploaded_name = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        excel_path = tmp.name

    if st.button("Convert to Shapefile"):
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(excel_path)
            else:
                df = pd.read_excel(excel_path)

            df.columns = [c.strip() for c in df.columns]

            polygons, names = [], []
            for _, row in df.iterrows():
                coords = []
                for i in range(1, 5):
                    lat = row.get(f"Lat{i}")
                    lon = row.get(f"Long{i}")
                    if pd.notna(lat) and pd.notna(lon):
                        # Apply transformation only if raw
                        if transform_choice == "Raw sheet (needs transformation)":
                            lat = transform_coord(lat)
                            lon = transform_coord(lon)
                        coords.append((lon, lat))
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
                    val = row["Temp code"]
                    if pd.notna(val):
                        try:
                            names.append(int(val))
                        except:
                            names.append(val)

            gdf = gpd.GeoDataFrame({"TempCode": names, "geometry": polygons}, crs="EPSG:4326")

            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = os.path.join(tmpdir, "output.shp")
                gdf.to_file(shp_path, driver="ESRI Shapefile")

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    for file in os.listdir(tmpdir):
                        if file.startswith("output."):
                            zipf.write(os.path.join(tmpdir, file), arcname=file)
                zip_buffer.seek(0)

                st.session_state.zip_bytes = zip_buffer.getvalue()
            st.success("✅ Shapefile created successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

st.header("Sentinel-2 Image Downloader")
st.write("Enter bounding box and date range to download a cloud-free 4-band Sentinel-2 image composite.")

with st.form("sentinel_form"):
    col1, col2 = st.columns(2)
    with col1:
        lat1 = st.number_input("Top Latitude", value=10.02)
        lon1 = st.number_input("Left Longitude", value=76.24)
    with col2:
        lat2 = st.number_input("Bottom Latitude", value=9.98)
        lon2 = st.number_input("Right Longitude", value=76.28)

    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    submit_btn = st.form_submit_button("Get Cloud-Free Composite")

if submit_btn:
    st.session_state["composite_image_bytes"] = None
    st.session_state["tiles_zip_bytes"] = None

    try:
        bounds = ee.Geometry.Rectangle([lon1, lat2, lon2, lat1])
        composite = build_composite(bounds, start_date, end_date)  # int16, 4 bands

        # ---- NEW: quick NoData advisory (coarse) ----
        try:
            frac_clear = clear_fraction_qa60(bounds, start_date, end_date, sample_scale=500).getInfo()
            if frac_clear is None:
                frac_clear = 0.0
            st.caption(f"Estimated clear coverage: ~{round(100*frac_clear,1)}% of AOI has at least one clear observation.")
            if frac_clear < 0.98:
                st.warning(
                    "Some pixels in this AOI may have no clear observations in the given date range. "
                    "The output GeoTIFF can contain NoData (black) in those areas. "
                    "Consider widening the date range for fuller coverage."
                )
        except Exception:
            # Non-blocking; continue with export even if this estimate fails
            pass

        # Decide: single file or tiled
        px_est = estimate_pixels(lat1, lat2, lon1, lon2, scale_m=10)
        # Very conservative: if > ~25–30M px, tile to avoid 50MB cap
        needs_tiling = px_est > 2.8e7

        with tempfile.TemporaryDirectory() as tmpdir:
            if not needs_tiling:
                # ---- Single export (small AOI) ----
                out_path = os.path.join(tmpdir, "sentinel_image.tif")
                geemap.ee_export_image(
                    composite,
                    filename=out_path,
                    scale=10,
                    region=bounds,
                    file_per_band=False,
                )
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.session_state["composite_image_bytes"] = f.read()
                    st.success("✅ Cloud-free composite image is ready for download!")
                else:
                    st.error("Image export failed. Try a smaller region or a longer date range.")
            else:
                # ---- Tiled export (big AOI): split and ZIP all tiles ----
                st.warning("Large AOI detected → exporting as multiple tiles and zipping them for download.")

                dlat, dlon = safe_tile_steps(lat1, lat2, lon1, lon2, scale_m=10, bands=4, bytes_per_sample=2, limit_mb=48)
                lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
                lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
                lat_edges = arange_edges(lat_min, lat_max, dlat)
                lon_edges = arange_edges(lon_min, lon_max, dlon)

                attempts = 0
                successes = 0
                failures = 0
                tile_files = []

                for yi in range(len(lat_edges) - 1):
                    for xi in range(len(lon_edges) - 1):
                        ymin, ymax = lat_edges[yi], lat_edges[yi + 1]
                        xmin, xmax = lon_edges[xi], lon_edges[xi + 1]
                        tile_geom = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
                        tile_name = f"tile_r{yi:03d}_c{xi:03d}.tif"
                        tile_path = os.path.join(tmpdir, tile_name)
                        attempts += 1
                        try:
                            geemap.ee_export_image(
                                composite,
                                filename=tile_path,
                                scale=10,
                                region=tile_geom,
                                file_per_band=False,
                            )
                            if os.path.exists(tile_path) and os.path.getsize(tile_path) > 0:
                                tile_files.append(tile_path)
                                successes += 1
                            else:
                                failures += 1
                        except Exception:
                            failures += 1

                st.info(f"Tiling complete. Attempts: {attempts}, succeeded: {successes}, failed: {failures}")

                if not tile_files:
                    st.error("All tile exports failed. Try shrinking the AOI or widening the date range.")
                else:
                    # Make a ZIP of all tiles
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fp in tile_files:
                            zf.write(fp, arcname=os.path.basename(fp))
                    zip_buffer.seek(0)
                    st.session_state["tiles_zip_bytes"] = zip_buffer.getvalue()
                    st.success("✅ Tiles are ready. Download the ZIP below.")

    except Exception as e:
        st.error(f"Error retrieving image: {e}")

# ---------------- Downloads ----------------
if st.session_state.get("composite_image_bytes") is not None:
    st.download_button(
        label="⬇️ Download Sentinel Image (GeoTIFF)",
        data=st.session_state["composite_image_bytes"],
        file_name="sentinel_image.tif",
        mime="image/tiff",
        key="dl_single",
    )

if st.session_state.get("tiles_zip_bytes") is not None:
    st.download_button(
        label="⬇️ Download Tiled Sentinel Image (ZIP)",
        data=st.session_state["tiles_zip_bytes"],
        file_name="sentinel_tiles.zip",
        mime="application/zip",
        key="dl_tiles",
    )

# Shapefile download (unchanged)
if st.session_state.get("zip_bytes") is not None:
    st.download_button(
        label="⬇️ Download Shapefile (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="shapefile_output.zip",
        mime="application/zip",
        key="download-btn"
    )
