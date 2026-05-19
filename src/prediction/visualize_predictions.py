"""Streamlit app to visualize deforestation probability predictions."""

import streamlit as st
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

st.set_page_config(page_title="Deforestation Risk Map", layout="wide")

st.title("Deforestation Risk Prediction Map")
st.markdown("Green = Low Risk | Red = High Risk")

# Sidebar for file selection
st.sidebar.header("Configuration")
data_dir = st.sidebar.text_input("Data directory", value=".")
model_dir = st.sidebar.text_input("Model output directory", value="model_outputs")

# Find available prediction files
model_path = Path(model_dir)
if model_path.exists():
    prediction_files = sorted(model_path.glob("p_loss_*.tif"))
    if prediction_files:
        selected_file = st.sidebar.selectbox(
            "Select prediction map",
            options=prediction_files,
            format_func=lambda x: x.name
        )
    else:
        st.sidebar.error("No prediction maps found (p_loss_*.tif)")
        st.stop()
else:
    st.error(f"Model directory not found: {model_path}")
    st.stop()

# Load the prediction raster
@st.cache_data
def load_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    return data, transform, crs

try:
    data, transform, crs = load_raster(selected_file)
except Exception as e:
    st.error(f"Error loading raster: {e}")
    st.stop()

# Get statistics
valid_data = data[np.isfinite(data)]
if len(valid_data) == 0:
    st.error("No valid data in raster")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Statistics")
st.sidebar.metric("Min Probability", f"{valid_data.min():.3f}")
st.sidebar.metric("Max Probability", f"{valid_data.max():.3f}")
st.sidebar.metric("Mean Probability", f"{valid_data.mean():.3f}")
st.sidebar.metric("Median Probability", f"{np.median(valid_data):.3f}")
st.sidebar.metric("Valid Pixels", f"{len(valid_data):,}")

# Threshold slider
threshold = st.sidebar.slider(
    "Risk threshold for highlighting",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Pixels above this probability will be shown as red"
)

# Create custom colormap: green to red
colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
n_bins = 100
cmap = LinearSegmentedColormap.from_list('green_red', colors, N=n_bins)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Main map with custom colormap
ax1 = axes[0]
im1 = ax1.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
ax1.set_title(f"Deforestation Risk Map\n{selected_file.name}", fontsize=14, fontweight='bold')
ax1.set_xlabel("Column")
ax1.set_ylabel("Row")
cbar1 = plt.colorbar(im1, ax=ax1, label="Loss Probability")
cbar1.set_label("Loss Probability", rotation=270, labelpad=20)

# Mask map showing high risk areas
ax2 = axes[1]
high_risk = np.where(np.isfinite(data), data >= threshold, np.nan)
im2 = ax2.imshow(high_risk, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
ax2.set_title(f"High Risk Areas (> {threshold:.2f})", fontsize=14, fontweight='bold')
ax2.set_xlabel("Column")
ax2.set_ylabel("Row")
cbar2 = plt.colorbar(im2, ax=ax2, label="Risk Level")

plt.tight_layout()
st.pyplot(fig)

# Statistics section
col1, col2, col3, col4 = st.columns(4)

high_risk_count = np.sum(valid_data >= threshold)
high_risk_pct = 100 * high_risk_count / len(valid_data)
medium_risk_count = np.sum((valid_data >= 0.3) & (valid_data < threshold))
low_risk_count = np.sum(valid_data < 0.3)

with col1:
    st.metric("High Risk Pixels", f"{high_risk_count:,}")
    st.metric("(% of valid)", f"{high_risk_pct:.1f}%")

with col2:
    st.metric("Medium Risk Pixels", f"{medium_risk_count:,}")
    st.metric("(0.3 - 0.5)", f"{100*medium_risk_count/len(valid_data):.1f}%")

with col3:
    st.metric("Low Risk Pixels", f"{low_risk_count:,}")
    st.metric("(< 0.3)", f"{100*low_risk_count/len(valid_data):.1f}%")

with col4:
    st.metric("CRS", str(crs) if crs else "Unknown")
    st.metric("Shape", f"{data.shape[0]:,} x {data.shape[1]:,}")

# Histogram
st.subheader("Probability Distribution")
fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
ax_hist.hist(valid_data, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax_hist.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
ax_hist.axvline(valid_data.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({valid_data.mean():.3f})')
ax_hist.set_xlabel("Loss Probability")
ax_hist.set_ylabel("Number of Pixels")
ax_hist.set_title("Distribution of Deforestation Probabilities")
ax_hist.legend()
ax_hist.grid(alpha=0.3)
st.pyplot(fig_hist)

# Quantile breakdown
st.subheader("Risk Quantiles")
quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
quantile_values = np.quantile(valid_data, quantiles)
quantile_df = pd.DataFrame({
    'Quantile': [f'{q*100:.0f}%' for q in quantiles],
    'Probability': quantile_values
})
st.dataframe(quantile_df, use_container_width=True)

# File info
st.sidebar.markdown("---")
st.sidebar.subheader("File Information")
st.sidebar.text(f"File: {selected_file.name}")
st.sidebar.text(f"Size: {selected_file.stat().st_size / 1024 / 1024:.1f} MB")
