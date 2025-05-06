import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from math import pow

# MUST be the first Streamlit command
st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Data Analytics Dashboard.csv", encoding_errors='ignore')
    df.columns = df.columns.str.strip().str.replace('\u202f', '').str.replace('\xa0', '')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    return df.dropna(subset=['Latitude', 'Longitude', 'Time'])

df = load_data()

# Header with institute emblem and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("csir-niist-neeri-logo.png", width=100)
with col2:
    st.title("🛰️ NatGEM National GHG Emission Inventory")

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
    st.success(f"Filtered data from **{start_date.date()}** to **{end_date.date()}**")
else:
    st.warning("Please select a valid date range.")
    st.stop()

# GHG selection
pollutants = ["Methane (PPM)", "CO (PPM)", "CO2 (PPM)", "VOC (PPB)"]
selected_pollutant = st.sidebar.selectbox("Select GHG for Interpolation & Time Series", pollutants)

# Satellite map with colored pin markers based on pollutant concentration
st.subheader("📍 Dumping Site")
m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=15, tiles=None)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri WorldImagery',
    overlay=False,
    control=True
).add_to(m)

# Categorize pollutant concentrations into low, moderate, high
quantiles = df[selected_pollutant].quantile([0.33, 0.66]).values
low_threshold, high_threshold = quantiles[0], quantiles[1]

def get_marker_color(value):
    if value <= low_threshold:
        return 'green'  # Low concentration
    elif value <= high_threshold:
        return 'yellow'  # Moderate concentration
    else:
        return 'red'  # High concentration

# Add pin markers with color coding
for _, row in df.iterrows():
    color = get_marker_color(row[selected_pollutant])
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=(f"<b>Time:</b> {row['Time']}<br><b>{selected_pollutant}:</b> {row[selected_pollutant]}"),
        icon=folium.Icon(
            color='white',
            icon_color=color,
            icon='map-marker',
            prefix='fa'
        )
    ).add_to(m)
    
    if color == 'yellow':
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=1.5,
            color='white',
            fill=True,
            fill_color='white',
            fill_opacity=1.0,
            popup=None
        ).add_to(m)

# Add a legend to the map
legend_html = f"""
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 90px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     ">
       <b>{selected_pollutant} Concentration (White Border)</b> <br>
       <i class="fa fa-map-marker" style="color:green"></i>  Low (≤ {low_threshold:.2f})<br>
       <i class="fa fa-map-marker" style="color:yellow"></i> <i class="fa fa-circle" style="color:white; font-size:8px;"></i> Moderate (≤ {high_threshold:.2f})<br>
       <i class="fa fa-map-marker" style="color:red"></i>  High (> {high_threshold:.2f})
     </div>
     """
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=700, height=500)

# IDW Interpolation
st.subheader(f"🌡️ IDW Interpolation Heatmap: {selected_pollutant}")

# Manual IDW interpolation function
def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x[:, None, None] - xi[None, :, :])**2 + (y[:, None, None] - yi[None, :, :])**2)
    weights = 1 / np.power(dist, power, where=dist!=0)
    weights[dist == 0] = 1e12  # Assign high weight to exact locations
    z_idw = np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)
    return z_idw

# Prepare data for interpolation
points = df[['Longitude', 'Latitude']].values
values = df[selected_pollutant].values
x, y = points[:, 0], points[:, 1]

# Define grid for interpolation
grid_lon = np.linspace(x.min(), x.max(), 15)  # Reduced grid size as per image
grid_lat = np.linspace(y.min(), y.max(), 15)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

# Perform IDW interpolation
z_idw = idw_interpolation(x, y, values, grid_x, grid_y)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the interpolated data
c = ax.contourf(grid_x, grid_y, z_idw, cmap="nipy_spectral_r", levels=100)
plt.colorbar(c, ax=ax, label=selected_pollutant)

# Add scatter points for actual data locations
ax.scatter(x, y, c='black', s=50, edgecolor='white')

ax.set_title(f"IDW Interpolated {selected_pollutant}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Save the figure to display in Streamlit
plt.savefig('idw_heatmap.png')
st.image('idw_heatmap.png')

# Time-series scatter plot
st.subheader(f"📈 Time-Series of {selected_pollutant}")
df_sorted = df.sort_values("Time")
fig2, ax2 = plt.subplots()
ax2.scatter(df_sorted['Time'], df_sorted[selected_pollutant], color='darkgreen', s=10)
ax2.set_title(f"Time-Series of {selected_pollutant}")
ax2.set_xlabel("Time")
ax2.set_ylabel(selected_pollutant)
fig2.autofmt_xdate()
st.pyplot(fig2)
