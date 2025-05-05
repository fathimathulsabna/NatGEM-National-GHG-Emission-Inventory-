import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from math import pow

# ✅ MUST be the first Streamlit command
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
    # Option 1: Use a local file (ensure the image file is in the same directory as this script)
    st.image("csir-niist-neeri-logo.png", width=100)
    
    # Option 2: Use a hosted image URL (uncomment and replace with the actual URL if preferred)
    # st.image("https://via.placeholder.com/100", width=100)
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

# 📍 Map with markers
st.subheader("📍 Monitoring Sites Map")
m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=15)
for _, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=(f"<b>Time:</b> {row['Time']}<br><b>{selected_pollutant}:</b> {row[selected_pollutant]}")
    ).add_to(m)
st_folium(m, width=700, height=500)

# 🔥 IDW Interpolation
def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x[:, None, None] - xi[None, :, :])**2 + (y[:, None, None] - yi[None, :, :])**2)
    weights = 1 / np.power(dist, power, where=dist!=0)
    weights[dist == 0] = 1e12  # assign high weight to exact locations
    z_idw = np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)
    return z_idw

st.subheader(f"🌡️ IDW Interpolation Heatmap: {selected_pollutant}")
points = df[['Longitude', 'Latitude']].values
values = df[selected_pollutant].values
x, y = points[:, 0], points[:, 1]

grid_lon = np.linspace(x.min(), x.max(), 100)
grid_lat = np.linspace(y.min(), y.max(), 100)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

z_idw = idw_interpolation(x, y, values, grid_x, grid_y)

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(grid_x, grid_y, z_idw, cmap="inferno", levels=100)
plt.colorbar(c, ax=ax, label=selected_pollutant)
ax.set_title(f"IDW Interpolated {selected_pollutant}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# 📊 Time-series plot
st.subheader(f"📈 Time-Series of {selected_pollutant}")
df_sorted = df.sort_values("Time")
fig2, ax2 = plt.subplots()
ax2.plot(df_sorted['Time'], df_sorted[selected_pollutant], color='darkgreen')
ax2.set_title(f"Time-Series of {selected_pollutant}")
ax2.set_xlabel("Time")
ax2.set_ylabel(selected_pollutant)
fig2.autofmt_xdate()
st.pyplot(fig2)