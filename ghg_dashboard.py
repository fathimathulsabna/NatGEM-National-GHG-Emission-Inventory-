import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from math import pow
import uuid
import matplotlib.dates as mdates
import os
import base64

st.set_page_config(page_title="NatGEM National GHG Emission Inventory", layout="wide")

# Clear all caches and set page config
st.session_state.clear()
st.cache_data.clear()
st.cache_resource.clear()

# Custom CSS to force side-by-side layout with no gap
st.markdown("""
<style>
/* Remove all gaps between components */
[data-testid="stHorizontalBlock"] {
    gap: 0px !important;
}
[data-testid="column"] {
    padding: 0px !important;
    margin: 0px !important;
}
/* Force maps to be exactly half width */
.map-column {
    width: 50% !important;
    min-width: 50% !important;
    max-width: 50% !important;
    padding: 0px !important;
    margin: 0px !important;
}
/* Remove container padding */
.stContainer {
    padding: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=60)  # Short cache timeout
def load_data():
    try:
        df = pd.read_csv("Data Analytics Dashboard.csv", encoding_errors='ignore')
        # Clean column names: strip whitespace, replace special characters
        df.columns = df.columns.str.strip().str.replace('\u202f', '').str.replace('\xa0', '').str.replace('°C', 'C')
        # Note: If 'Time' column has a consistent format (e.g., 'YYYY-MM-DD'), specify it to avoid warning, e.g., pd.to_datetime(df['Time'], format='%Y-%m-%d')
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        
        # Define required columns (excluding Temperature for now)
        required_columns = ['site', 'Latitude', 'Longitude', 'Time', 'Methane (PPM)', 'CO (PPM)', 'CO2 (PPM)', 'VOC (PPB)']
        # Normalize column names for comparison (case-insensitive)
        available_columns = [col.lower() for col in df.columns]
        required_columns_lower = [col.lower() for col in required_columns]
        
        # Check for missing required columns
        missing_columns = [col for col in required_columns_lower if col not in available_columns]
        if missing_columns:
            st.error(f"Error: Missing required columns in the dataset: {', '.join(missing_columns)}")
            st.write("Available columns in the dataset:", list(df.columns))
            return pd.DataFrame()
        
        # Map normalized columns back to original names
        column_mapping = {col.lower(): col for col in df.columns}
        df.columns = [column_mapping[col.lower()] for col in df.columns]
        
        # Check for optional Temperature column
        temperature_col = None
        for col in df.columns:
            if 'temperature' in col.lower() and 'c' in col.lower():
                temperature_col = col
                break
        
        if temperature_col:
            df = df.rename(columns={temperature_col: 'Temperature (C)'})
        
        return df.dropna(subset=['Latitude', 'Longitude', 'Time'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No data loaded. Please check the dataset and try again.")
    st.stop()

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Get the current working directory and image paths
cwd = os.getcwd()
img_path_1 = os.path.join(cwd, "images/csir-neeri-logo.png")
img_path_2 = os.path.join(cwd, "images/csir-nal-logo.png")
img_path_3 = os.path.join(cwd, "images/csir-niist-logo.jpeg")

# Convert images to base64
img_base64_1 = get_base64_image(img_path_1)
img_base64_2 = get_base64_image(img_path_2)
img_base64_3 = get_base64_image(img_path_3)

# Use base64 strings in the HTML
st.markdown(f"""
    <style>
        .header-container {{
            background-color: #d6f5d6;
            padding: 20px;
            border-radius: 8px;
        }}
        .logo-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .logo-group {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .header-title {{
            font-size: 28px;
            font-weight: bold;
            color: #134b13;
            text-align: center;
            flex-grow: 1;
            margin-left: 20px;
        }}
    </style>

    <div class="header-container">
        <div class="logo-row">
            <div class="logo-group">
                <img src="data:image/png;base64,{img_base64_1}" width="90" alt="CSIR NEERI">
                <img src="data:image/png;base64,{img_base64_2}" width="90" alt="CSIR NAL">
                <img src="data:image/jpeg;base64,{img_base64_3}" width="100" alt="CSIR NIIST">
            </div>
            <div class="header-title">🛰️ NatGEM National GHG Emission Inventory</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔧 Filter Options")
sites = df['site'].dropna().unique().tolist()
selected_site = st.sidebar.selectbox("Select Site", sites)
df = df[df['site'] == selected_site]

# Date range
min_date = df["Time"].min().date()
max_date = df["Time"].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["Time"] >= start_date) & (df["Time"] <= end_date)]
else:
    st.warning("Please select a valid date range.")
    st.stop()

# GHG selection
pollutants = ["Methane (PPM)", "CO (PPM)", "CO2 (PPM)", "VOC (PPB)"]
selected_pollutant = st.sidebar.selectbox("Select GHG for Interpolation & Pie Chart", pollutants)

# Combined maps section
st.subheader("📍 Dumping Site & 🌡️ IDW Interpolation Heatmap")

# Create container with columns
map_container = st.container()
col1, col2 = map_container.columns(2)

# Satellite Map (Left)
with col1:
    m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=15, tiles=None, max_zoom=23, control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri WorldImagery',
        overlay=False,
        control=True,
        max_zoom=23
    ).add_to(m)

    # Categorize concentrations
    quantiles = df[selected_pollutant].quantile([0.33, 0.66]).values
    low_threshold, high_threshold = quantiles[0], quantiles[1]
    medium_value = (low_threshold + high_threshold) / 2  # Calculate medium concentration as average of low and high thresholds

    def get_marker_color(value):
        if value <= low_threshold: return 'green'
        elif value <= high_threshold: return 'yellow'
        else: return 'red'

    # Add markers
    for _, row in df.iterrows():
        color = get_marker_color(row[selected_pollutant])
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"<b>Time:</b> {row['Time']}<br><b>{selected_pollutant}:</b> {row[selected_pollutant]}",
            icon=folium.Icon(color='white', icon_color=color, icon='map-marker', prefix='fa')
        ).add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 110px; 
    background-color: black; border:2px solid grey; z-index:9999; font-size:14px; color: white;">
      <b>{selected_pollutant} Concentration</b> <br>
      <i class="fa fa-map-marker" style="color:green"></i> Low (≤ {low_threshold:.2f})<br>
      <i class="fa fa-map-marker" style="color:yellow"></i> Moderate ({medium_value:.2f})<br>
      <i class="fa fa-map-marker" style="color:red"></i> High (> {high_threshold:.2f})
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add zoom control and lat/lng popup to check zoom level
    folium.LatLngPopup().add_to(m)  # Shows lat/lng and zoom level on click

    st_folium(m, width=500, height=500, key="satellite_map")

# IDW Interpolation (Right)
with col2:
    def idw_interpolation(x, y, z, xi, yi, power=2):
        dist = np.sqrt((x[:, None, None] - xi[None, :, :])**2 + (y[:, None, None] - yi[None, :, :])**2)
        weights = 1 / np.power(dist, power, where=dist != 0)
        weights[dist == 0] = 1e12
        return np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)

    points = df[['Longitude', 'Latitude']].values
    values = df[selected_pollutant].values
    x, y = points[:, 0], points[:, 1]

    # Add padding to ensure all points are visible
    lon_padding = (x.max() - x.min()) * 0.1  # 10% of longitude range
    lat_padding = (y.max() - y.min()) * 0.1  # 10% of latitude range
    grid_lon = np.linspace(x.min() - lon_padding, x.max() + lon_padding, 15)
    grid_lat = np.linspace(y.min() - lat_padding, y.max() + lat_padding, 15)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    z_idw = idw_interpolation(x, y, values, grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(5, 5))
    c = ax.contourf(grid_x, grid_y, z_idw, cmap="nipy_spectral_r", levels=100)
    plt.colorbar(c, ax=ax, label=selected_pollutant)
    ax.scatter(x, y, c='black', s=50, edgecolor='white')
    ax.set_title(f"IDW Interpolated {selected_pollutant}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# GHG Flux Interpolation Map and Pie Chart
st.subheader(f"🌡️ GHG Flux Interpolation Map & 📊 Emission of {selected_pollutant}")

# Calculate GHG flux in mg/m²/min
# Step 1: Convert concentration to mg/m³
concentration_to_mass = {
    "Methane (PPM)": 0.655,  # 1 PPM = 0.655 mg/m³ (CH₄, molar mass 16 g/mol)
    "CO (PPM)": 1.15,        # 1 PPM = 1.15 mg/m³ (CO, molar mass 28 g/mol)
    "CO2 (PPM)": 1.8,        # 1 PPM = 1.8 mg/m³ (CO₂, molar mass 44 g/mol)
    "VOC (PPB)": 0.004       # 1 PPB = 0.004 mg/m³ (assuming avg molar mass ~100 g/mol)
}
# Step 2: Convert mg/m³ to mg/m²/min using a flux rate factor (placeholder: 0.01 m³/m²/min)
flux_rate_factor = 0.01  # m³/m²/min (example value)

# Calculate concentration in mg/m³
df['Concentration (mg/m³)'] = df[selected_pollutant] * concentration_to_mass[selected_pollutant]
# Calculate flux in mg/m²/min
df['GHG_Flux'] = df['Concentration (mg/m³)'] * flux_rate_factor

# Create container for flux map and pie chart
flux_pie_container = st.container()
flux_col, pie_col = flux_pie_container.columns(2)  # Two columns for side-by-side display

# Generate flux map for the selected site
def create_flux_map(site_data, site_name, selected_pollutant):
    points = site_data[['Longitude', 'Latitude']].values
    values = site_data['GHG_Flux'].values
    x, y = points[:, 0], points[:, 1]

    # Add padding to ensure all points are visible
    lon_padding = (x.max() - x.min()) * 0.1  # 10% of longitude range
    lat_padding = (y.max() - y.min()) * 0.1  # 10% of latitude range
    grid_lon = np.linspace(x.min() - lon_padding, x.max() + lon_padding, 15)
    grid_lat = np.linspace(y.min() - lat_padding, y.max() + lat_padding, 15)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    z_idw = idw_interpolation(x, y, values, grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(5, 5))
    c = ax.contourf(grid_x, grid_y, z_idw, cmap="nipy_spectral_r", levels=100)
    plt.colorbar(c, ax=ax, label=f"{selected_pollutant} Flux (mg/m²/min)")
    ax.scatter(x, y, c='black', s=50, edgecolor='white')
    ax.set_title(f"IDW Interpolated {selected_pollutant} Flux - {site_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig

# Display flux map
site_data = df[df['site'] == selected_site]
with flux_col:
    if not site_data.empty:
        fig_flux = create_flux_map(site_data, selected_site, selected_pollutant)
        st.pyplot(fig_flux, use_container_width=True)
        plt.close(fig_flux)
    else:
        st.warning(f"No data available for {selected_site}.")

# Pie chart
with pie_col:
    low_count = len(df[df[selected_pollutant] <= low_threshold])
    moderate_count = len(df[(df[selected_pollutant] > low_threshold) & (df[selected_pollutant] <= high_threshold)])
    high_count = len(df[df[selected_pollutant] > high_threshold])

    labels = ['Low', 'Moderate', 'High']
    sizes = [low_count, moderate_count, high_count]
    colors = ['green', 'yellow', 'red']
    explode = (0.05, 0.05, 0.05)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12})
    ax2.axis('equal')
    ax2.set_title(f"{selected_pollutant} Emission Levels", fontsize=14, pad=10)

    plt.legend(
        labels=[
            f"Low (≤ {low_threshold:.2f})",
            f"Moderate (≤ {high_threshold:.2f})",
            f"High (> {high_threshold:.2f})"
        ],
        loc="best",
        fontsize=10
    )

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# Average Values Table
st.subheader("📊 Average Values with time at the location")

# Calculate averages for the selected site
avg_temperature = df['Temperature (C)'].mean() if 'Temperature (C)' in df.columns else "N/A"
avg_methane = df['Methane (PPM)'].mean()
avg_co2 = df['CO2 (PPM)'].mean()
avg_voc = df['VOC (PPB)'].mean()

# Calculate GHG Emission Factor based on the selected pollutant
scaling_factors = {
    "Methane (PPM)": 0.5,  # Example: 0.5 g/hr per PPM
    "CO (PPM)": 0.4,       # Example: 0.4 g/hr per PPM
    "CO2 (PPM)": 0.3,      # Example: 0.3 g/hr per PPM
    "VOC (PPB)": 0.1       # Example: 0.1 g/hr per PPB
}
avg_ghg_value = df[selected_pollutant].mean()
ghg_emission_factor = avg_ghg_value * scaling_factors[selected_pollutant]

# Create a DataFrame for the table
avg_data = {
    "City": [f"{selected_site}, Delhi"],
    "Temperature (C)": [f"{avg_temperature:.1f}" if avg_temperature != "N/A" else "N/A"],
    "GHG Emission Factor (g/hr)": [f"{ghg_emission_factor:.1f}"],
    "Methane (PPM)": [f"{avg_methane:.1f}"],
    "CO2 (PPM)": [f"{avg_co2:.1f}"],
    "VOC (PPB)": [f"{avg_voc:.1f}"]
}
avg_df = pd.DataFrame(avg_data)

# Display the table
st.dataframe(
    avg_df,
    use_container_width=True,
    hide_index=True,
)

st.subheader(f"📈 {selected_pollutant} Concentration Over Time")

# Sort data by Time for plotting
df_time = df.sort_values("Time")

plt.figure(figsize=(12, 5))
plt.plot(df_time["Time"], df_time[selected_pollutant], marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel(f"{selected_pollutant} Concentration")
plt.title(f"{selected_pollutant} Concentration Over Time at {selected_site}")
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)

# Format x-axis dates to show only date (no 00:00)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

st.pyplot(plt)
plt.close()

# Note about missing temperature data
if avg_temperature == "N/A":
    st.markdown("*Note: Temperature data is not available in the current dataset.*")
