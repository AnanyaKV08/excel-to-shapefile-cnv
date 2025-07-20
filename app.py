import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import tempfile
import zipfile
import os
from io import BytesIO

st.title("Excel to shapefile converter")
st.write("Upload your Excel file, and download a shapefile with polygons.")

# Initialize session state variables
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

# Detect a *new* upload and clear previous state
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_name:
        st.session_state.zip_bytes = None
        st.session_state.last_uploaded_name = uploaded_file.name

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        excel_path = tmp.name

    if st.button("Convert to Shapefile"):
        try:
            df = pd.read_excel(excel_path)
            df.columns = [c.strip() for c in df.columns]

            polygons, names = [], []
            for _, row in df.iterrows():
                coords = []
                for i in range(1, 5):
                    lat = row.get(f"Lat{i}")
                    lon = row.get(f"Long{i}")
                    if pd.notna(lat) and pd.notna(lon):
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

            # Save to zip in memory
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
            st.success("Shapefile created successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

# Show download button if zip bytes exist
if st.session_state.zip_bytes is not None:
    st.download_button(
        label="⬇️ Download Shapefile (ZIP)",
        data=st.session_state.zip_bytes,
        file_name="shapefile_output.zip",
        mime="application/zip",
        key="download-btn" 
    )
