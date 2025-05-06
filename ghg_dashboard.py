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
    try:
        df = pd.read_csv("Data Analytics Dashboard.csv", encoding_errors='ignore')
        df.columns = df.columns.str.strip().str.replace('\u202f', '').str.replace('\xa0', '')
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        required_columns = ['site', 'Latitude', 'Longitude', 'Time', 'Methane (PPM)', 'CO (PPM)', 'CO2 (PPM)', 'VOC (PPB)']
        if not all(col in df.columns for col in required_columns):
            st.error("Error: Missing required columns in the dataset.")
            return pd.DataFrame()
        return df.dropna(subset=['Latitude', 'Longitude', 'Time'])
    except FileNotFoundError:
        st.error("Error: 'Data Analytics Dashboard.csv' file not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No data loaded. Please check the dataset and try again.")
    st.stop()

# Header with institute emblem and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("csir-niist-neeri-logo.png", width=100)
with col2:
    st.title("🛰️ NatGEM National GHG Emission Inventory")

# Sidebar filters
st.sidebar.header("🔧 Filter Options")
sites = df['site'].dropna().unique().tolist()
if not sites:
    st.error("No valid sites found in the dataset.")
    st.stop()
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
selected_pollutant = st.sidebar.selectbox("Select GHG for Interpolation & Pie Chart", pollutants)

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
        return 'green'
    elif value <= high_threshold:
        return 'yellow'
    else:
        return 'red'

# Add pin markers
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

# Legend
legend_html = f"""
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 90px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     ">
       <b>{selected_pollutant} Concentration (White Border)</b> <br>
       <i class="fa fa-map-marker" style="color:green"></i> Low (≤ {low_threshold:.2f})<br>
       <i class="fa fa-map-marker" style="color:yellow"></i> <i class="fa fa-circle" style="color:white; font-size:8px;"></i> Moderate (≤ {high_threshold:.2f})<br>
       <i class="fa fa-map-marker" style="color:red"></i> High (> {high_threshold:.2f})
     </div>
     """
m.get_root().html.add_child(folium.Element(legend_html))
st_folium(m, width=700, height=500)

# IDW Interpolation
st.subheader(f"🌡️ IDW Interpolation Heatmap: {selected_pollutant}")

def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x[:, None, None] - xi[None, :, :])**2 + (y[:, None, None] - yi[None, :, :])**2)
    weights = 1 / np.power(dist, power, where=dist != 0)
    weights[dist == 0] = 1e12
    z_idw = np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)
    return z_idw

points = df[['Longitude', 'Latitude']].values
values = df[selected_pollutant].values
x, y = points[:, 0], points[:, 1]

grid_lon = np.linspace(x.min(), x.max(), 15)
grid_lat = np.linspace(y.min(), y.max(), 15)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

z_idw = idw_interpolation(x, y, values, grid_x, grid_y)

fig, ax = plt.subplots(figsize=(7, 5))
c = ax.contourf(grid_x, grid_y, z_idw, cmap="nipy_spectral_r", levels=100)
plt.colorbar(c, ax=ax, label=selected_pollutant)
ax.scatter(x, y, c='black', s=50, edgecolor='white')
ax.set_title(f"IDW Interpolated {selected_pollutant}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig('idw_heatmap.png', bbox_inches='tight', dpi=100)
st.image('idw_heatmap.png', width=700)
plt.close(fig)

# Pie chart
st.subheader(f"📊 Emission of {selected_pollutant}")
low_count = len(df[df[selected_pollutant] <= low_threshold])
moderate_count = len(df[(df[selected_pollutant] > low_threshold) & (df[selected_pollutant] <= high_threshold)])
high_count = len(df[df[selected_pollutant] > high_threshold])

labels = ['Low', 'Moderate', 'High']
sizes = [low_count, moderate_count, high_count]
colors = ['green', 'yellow', 'red']
explode = (0.05, 0.05, 0.05)

fig2, ax2 = plt.subplots(figsize=(7, 5))
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
plt.savefig('pie_chart.png', bbox_inches='tight', dpi=100)
st.image('pie_chart.png', width=700)
plt.close(fig2)
