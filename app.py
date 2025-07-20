import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import tempfile
import zipfile
import os
from io import BytesIO

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

if st.session_state.zip_bytes is not None:
    st.download_button(
        label="⬇️ Download Shapefile (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="shapefile_output.zip",
        mime="application/zip",
        key="download-btn"
    )
