import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from datetime import datetime
from math import pow
import uuid
import matplotlib.dates as mdates
import os
import base64
import plotly.express as px

st.set_page_config(page_title="NatGEM National GHG Emission Inventory", layout="wide")

# Clear all caches and set page config
st.session_state.clear()
st.cache_data.clear()
st.cache_resource.clear()

# Custom CSS to force side-by-side layout with no gap and style the table
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
/* Style the dataframe table with light green background */
.stDataFrame table {
    width: 100%;
    border-collapse: collapse;
}
.stDataFrame th, .stDataFrame td {
    background-color: #d6f5d6 !important;
    border: 1px solid #ccc !important;
    padding: 8px !important;
    text-align: center !important;
}
.stDataFrame th {
    font-weight: bold !important;
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

# Get the current working directory and image paths (logos in same directory as script)
cwd = os.getcwd()
img_path_1 = os.path.join(cwd, "csir-neeri-logo.png")
img_path_2 = os.path.join(cwd, "csir-nal-logo.png")
img_path_3 = os.path.join(cwd, "csir-niist-logo.jpeg")

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

# Calculate thresholds for pie chart and satellite map
quantiles = df[selected_pollutant].quantile([0.33, 0.66]).values
low_threshold, high_threshold = quantiles[0], quantiles[1]
medium_value = (low_threshold + high_threshold) / 2  # Calculate medium concentration

# Dumping Site Map and Pie Chart
st.subheader(f"📍 Dumping Site & 📊 Emission of {selected_pollutant}")

# Create container for satellite map and pie chart
map_pie_container = st.container()
col1, pie_col = map_pie_container.columns(2)

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

# Interactive Pie Chart (Right)
with pie_col:
    low_count = len(df[df[selected_pollutant] <= low_threshold])
    moderate_count = len(df[(df[selected_pollutant] > low_threshold) & (df[selected_pollutant] <= high_threshold)])
    high_count = len(df[df[selected_pollutant] > high_threshold])

    labels = ['Low', 'Moderate', 'High']
    sizes = [low_count, moderate_count, high_count]
    colors = ['green', 'yellow', 'red']

    # Filter out zero-count categories
    valid_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
    if valid_data:
        labels, sizes, colors = zip(*valid_data)
        pie_data = pd.DataFrame({
            'Category': labels,
            'Count': sizes,
            'Threshold': [f"≤ {low_threshold:.2f}" if label == 'Low' else 
                          f"≤ {high_threshold:.2f}" if label == 'Moderate' else 
                          f">#{high_threshold:.2f}" for label in labels]
        })

        fig2 = px.pie(
            pie_data,
            names='Category',
            values='Count',
            title=None,
            color_discrete_sequence=colors
        )
        fig2.update_traces(
            textinfo='percent+label',
            pull=[0.05] * len(sizes),
            textfont=dict(size=12),
            customdata=pie_data[['Threshold']],
            hovertemplate='%{label}: %{value} (%{percent})<br>Threshold: %{customdata[0]}'
        )
        fig2.update_layout(
            legend=dict(
                title=f"{selected_pollutant} Thresholds",
                font=dict(size=10)
            ),
            margin=dict(t=50, b=50, l=50, r=50),
            annotations=[dict(
                text=f"Total: {sum(sizes)}",
                x=0.5,
                y=-0.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12)
            )]
        )
        st.plotly_chart(fig2, width=550, height=550)  # Slightly larger pie chart
    else:
        st.warning("No data available for the pie chart.")

# Interpolation Heatmaps
st.subheader("🌡️ Interpolation Heatmaps")

# Create container for IDW and flux interpolation maps
interp_container = st.container()
col2, flux_col = interp_container.columns(2)

# IDW Interpolation (Left)
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
    
    # Format x-axis (Longitude) as degrees with E/W
    def lon_formatter(x, pos):
        return f"{abs(x):.3f}°{'E' if x >= 0 else 'W'}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lon_formatter))
    ax.set_xlabel("Longitude")

    # Format y-axis (Latitude) as degrees with N/S
    def lat_formatter(y, pos):
        return f"{abs(y):.3f}°{'N' if y >= 0 else 'S'}"
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lat_formatter))
    ax.set_ylabel("Latitude")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# GHG Flux Interpolation Map (Right)
# Calculate GHG flux in mg/m²/min
concentration_to_mass = {
    "Methane (PPM)": 0.655,  # 1 PPM = 0.655 mg/m³ (CH₄, molar mass 16 g/mol)
    "CO (PPM)": 1.15,        # 1 PPM = 1.15 mg/m³ (CO, molar mass 28 g/mol)
    "CO2 (PPM)": 1.8,        # 1 PPM = 1.8 mg/m³ (CO₂, molar mass 44 g/mol)
    "VOC (PPB)": 0.004       # 1 PPB = 0.004 mg/m³ (assuming avg molar mass ~100 g/mol)
}
flux_rate_factor = 0.01  # m³/m²/min (example value)
df['Concentration (mg/m³)'] = df[selected_pollutant] * concentration_to_mass[selected_pollutant]
df['GHG_Flux'] = df['Concentration (mg/m³)'] * flux_rate_factor

# Generate flux map
def create_flux_map(site_data, site_name, selected_pollutant):
    points = site_data[['Longitude', 'Latitude']].values
    values = site_data['GHG_Flux'].values
    x, y = points[:, 0], points[:, 1]

    # Add padding to ensure all points are visible
    lon_padding = (x.max() - x.min()) * 0.1
    lat_padding = (y.max() - y.min()) * 0.1
    grid_lon = np.linspace(x.min() - lon_padding, x.max() + lon_padding, 15)
    grid_lat = np.linspace(y.min() - lat_padding, y.max() + lat_padding, 15)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    z_idw = idw_interpolation(x, y, values, grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(5, 5))
    c = ax.contourf(grid_x, grid_y, z_idw, cmap="nipy_spectral_r", levels=100)
    plt.colorbar(c, ax=ax, label=f"{selected_pollutant} Flux (mg/m²/min)")
    ax.scatter(x, y, c='black', s=50, edgecolor='white')
    ax.set_title(f"IDW Interpolated {selected_pollutant} Flux - {site_name}")

    # Format x-axis (Longitude) as degrees with E/W
    def lon_formatter(x, pos):
        return f"{abs(x):.3f}°{'E' if x >= 0 else 'W'}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lon_formatter))
    ax.set_xlabel("Longitude")

    # Format y-axis (Latitude) as degrees with N/S
    def lat_formatter(y, pos):
        return f"{abs(y):.3f}°{'N' if y >= 0 else 'S'}"
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lat_formatter))
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

# Display the table with light green styling
st.dataframe(
    avg_df,
    use_container_width=True,
    hide_index=True,
)

# Display average temperature
st.markdown(f"**Average Temperature at {selected_site}:** {avg_temperature:.1f}°C" if avg_temperature != "N/A" else "**Average Temperature:** N/A")

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
