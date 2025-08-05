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
from rasterio.merge import merge as merge_rasters
import rasterio
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

# Transformation helper
def transform_coord(value):
    step1 = value / 100.0
    step2 = int(step1)
    step3 = step1 - step2
    step4 = step3 * 100
    step5 = step4 * 100 / 60
    step6 = step5 / 100
    step7 = step6 + step2
    return step7

st.title("Excel to shapefile converter")
st.write("Upload your Excel file, choose whether data is raw or already transformed, and download a shapefile.")

# Initialize session state variables
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "composite_image_bytes" not in st.session_state:
    st.session_state["composite_image_bytes"] = None


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
    st.session_state["composite_image_bytes"] = None  # Reset previous output

    try:
        bounds = ee.Geometry.Rectangle([lon1, lat2, lon2, lat1])  # Left, Bottom, Right, Top

        def mask_clouds(image):
            qa = image.select("QA60")
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0)
            )
            return image.updateMask(mask).copyProperties(image, ["system:time_start"])

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(bounds)
            .filterDate(str(start_date), str(end_date))
            .map(mask_clouds)
        )

        count = collection.size().getInfo()

        if count == 0:
            st.error("No suitable cloud-free pixels found in the given range and region.")
        else:
            st.write(f"Found {count} valid (masked) images in the date range.")
            with st.spinner("Processing image for download..."):
                def should_tile(lat1, lat2, lon1, lon2):
                    # Rough EE export limit: ~50km x 50km at 10m resolution
                    return abs(lat1 - lat2) > 0.45 or abs(lon1 - lon2) > 0.45

                composite = collection.select(["B4", "B3", "B2", "B8"]).median()

                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = os.path.join(tmpdir, "sentinel_image.tif")

                    if not should_tile(lat1, lat2, lon1, lon2):
                        # Normal export
                        geemap.ee_export_image(
                            composite,
                            filename=out_path,
                            scale=10,
                            region=bounds,
                            file_per_band=False,
                        )
                    else:
                        st.warning("⚠️ Large area detected. Processing as a tiled composite...")

                        step = 0.2 
                        tiles = []
                        for lat_start in np.arange(min(lat1, lat2), max(lat1, lat2), step):
                            for lon_start in np.arange(min(lon1, lon2), max(lon1, lon2), step):
                                tile_bounds = ee.Geometry.Rectangle([
                                    lon_start,
                                    lat_start,
                                    min(lon_start + step, max(lon1, lon2)),
                                    min(lat_start + step, max(lat1, lat2))
                                ])
                                tile_path = os.path.join(tmpdir, f"tile_{lat_start}_{lon_start}.tif")
                                try:
                                    geemap.ee_export_image(
                                        composite,
                                        filename=tile_path,
                                        scale=10,
                                        region=tile_bounds,
                                        file_per_band=False,
                                    )
                                    if os.path.exists(tile_path):
                                        tiles.append(tile_path)
                                except Exception as e:
                                    st.warning(f"Tile {lat_start:.2f},{lon_start:.2f} failed: {e}")

                        if not tiles:
                            st.error("Image export failed. No tiles were generated.")
                        else:

                            # Merge tiles
                            src_files_to_mosaic = [rasterio.open(tp) for tp in tiles]
                            mosaic, out_trans = merge_rasters(src_files_to_mosaic)

                            out_meta = src_files_to_mosaic[0].meta.copy()
                            out_meta.update({
                                "driver": "GTiff",
                                "height": mosaic.shape[1],
                                "width": mosaic.shape[2],
                                "transform": out_trans,
                            })

                            with rasterio.open(out_path, "w", **out_meta) as dest:
                                dest.write(mosaic)

                    # Read result for download
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            st.session_state["composite_image_bytes"] = f.read()
                    else:
                        st.error("Image export failed. Try a smaller region or larger scale value.")

    except Exception as e:
        st.error(f"Error retrieving image: {e}")

if st.session_state.get("composite_image_bytes") is not None:
    st.download_button(
        label="⬇️ Download Sentinel Image (GeoTIFF)",
        data=st.session_state["composite_image_bytes"],
        file_name="sentinel_image.tif",
        mime="image/tiff"
    )
    st.success("✅ Cloud-free composite image is ready for download!")

# Keep shapefile download section as-is if using both tools in same app
if st.session_state.get("zip_bytes") is not None:
    st.download_button(
        label="⬇️ Download Shapefile (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="shapefile_output.zip",
        mime="application/zip",
        key="download-btn"
    )
