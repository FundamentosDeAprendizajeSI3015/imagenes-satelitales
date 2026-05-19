"""Visualize model predictions overlaid on satellite imagery."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

BAND_NAMES = ["red", "nir", "swir1", "swir2", "ndvi"]


def load_model_and_features(model_path: Path, features_path: Path):
    """Load trained model and feature list."""
    print(f"[VIZ] Loading model from {model_path.name}...", flush=True)
    model = joblib.load(model_path)
    features = json.loads(features_path.read_text())
    return model, features


def make_features_from_window(data_t: np.ndarray, rows_abs: np.ndarray, 
                              cols_abs: np.ndarray, year_t: int, 
                              year_tp1: int, src_t):
    """Create feature dict from spectral data (simplified from build_supervised_dataset.py)."""
    from build_supervised_dataset import make_features, safe_index
    
    red, nir, swir1, swir2, ndvi = data_t
    ndmi = safe_index(nir - swir1, nir + swir1)
    nbr = safe_index(nir - swir2, nir + swir2)
    swir_ratio = safe_index(swir1, swir2)
    
    features = {
        "year_t": np.full(rows_abs.shape, year_t, dtype=np.int16),
        "year_tp1": np.full(rows_abs.shape, year_tp1, dtype=np.int16),
        "interval_years": np.full(rows_abs.shape, year_tp1 - year_t, dtype=np.int16),
        "red_t": red,
        "nir_t": nir,
        "swir1_t": swir1,
        "swir2_t": swir2,
        "ndvi_t": ndvi,
        "ndmi_t": ndmi,
        "nbr_t": nbr,
        "swir1_swir2_ratio_t": swir_ratio,
        "prev_interval_years": np.full(rows_abs.shape, -1, dtype=np.int16),
        "ndvi_prev": np.full(rows_abs.shape, np.nan, dtype=np.float32),
        "delta_ndvi_prev_to_t": np.full(rows_abs.shape, np.nan, dtype=np.float32),
        "nbr_prev": np.full(rows_abs.shape, np.nan, dtype=np.float32),
        "delta_nbr_prev_to_t": np.full(rows_abs.shape, np.nan, dtype=np.float32),
    }
    return features


def predict_on_raster(raster_path: Path, model, features_list: list,
                     year_t: int, year_tp1: int, 
                     forest_ndvi_threshold: float = 0.65):
    """Make predictions on full raster."""
    print(f"[VIZ] Opening raster {raster_path.name}...", flush=True)
    
    with rasterio.open(raster_path) as src_t:
        profile = src_t.profile.copy()
        height, width = src_t.height, src_t.width
        
        # Output array for probabilities
        predictions = np.full((height, width), np.nan, dtype=np.float32)
        ndvi_map = np.full((height, width), np.nan, dtype=np.float32)
        
        pixel_count = 0
        predicted_count = 0
        
        print(f"[VIZ] Processing raster ({height} x {width})...", flush=True)
        
        for _, window in src_t.block_windows(1):
            data_t = src_t.read(window=window, masked=False).astype(np.float32)
            
            # Check validity
            valid = np.all(np.isfinite(data_t), axis=0) & (data_t[4] >= forest_ndvi_threshold)
            
            if valid.any():
                flat_idx = np.flatnonzero(valid)
                local_rows, local_cols = np.unravel_index(flat_idx, (window.height, window.width))
                rows_abs = local_rows.astype(np.int64) + window.row_off
                cols_abs = local_cols.astype(np.int64) + window.col_off
                
                # Extract features
                local_arrays = {name: data_t[idx] for idx, name in enumerate(BAND_NAMES)}
                selected_features = {
                    key: value.reshape(-1)[flat_idx]
                    for key, value in local_arrays.items()
                }
                data_t_selected = np.vstack([selected_features[name] for name in BAND_NAMES]).astype(np.float32)
                
                # Create feature dict
                feature_dict = make_features_from_window(data_t_selected, rows_abs, cols_abs, 
                                                        year_t, year_tp1, src_t)
                df = pd.DataFrame(feature_dict)
                
                # Predict
                proba = model.predict_proba(df[features_list])[:, 1].astype(np.float32)
                
                # Write to output arrays
                predictions[rows_abs, cols_abs] = proba
                ndvi_map[rows_abs, cols_abs] = data_t[4].reshape(-1)[flat_idx]
                
                predicted_count += len(proba)
            
            pixel_count += window.height * window.width
        
        print(f"[VIZ] Predicted {predicted_count:,} forest pixels", flush=True)
        
    return predictions, ndvi_map, profile


def visualize_predictions(predictions: np.ndarray, satellite_data: np.ndarray,
                         out_dir: Path, year_t: int, year_tp1: int):
    """Create comprehensive visualization with predictions overlaid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Satellite false color (NIR, Red, SWIR1)
    ax1 = fig.add_subplot(gs[0, 0])
    false_color = np.stack([satellite_data[1], satellite_data[0], satellite_data[2]], axis=-1)
    false_color = np.clip(false_color / 0.3, 0, 1)
    ax1.imshow(false_color)
    ax1.set_title(f'Satellite False Color {year_t}', fontweight='bold')
    ax1.axis('off')
    
    # 2. NDVI map
    ax2 = fig.add_subplot(gs[0, 1])
    ndvi = satellite_data[4]
    im_ndvi = ax2.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title(f'NDVI {year_t}', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im_ndvi, ax=ax2, label='NDVI')
    
    # 3. Deforestation probability (full range)
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
    cmap_prob = LinearSegmentedColormap.from_list('risk', colors, N=100)
    im_prob = ax3.imshow(predictions, cmap=cmap_prob, vmin=0, vmax=1)
    ax3.set_title(f'Deforestation Risk (All probabilities)', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im_prob, ax=ax3, label='Loss Probability')
    
    # 4. High-risk pixels (threshold 0.3)
    ax4 = fig.add_subplot(gs[1, 0])
    high_risk = np.where(np.isfinite(predictions), predictions >= 0.3, np.nan)
    im_high = ax4.imshow(high_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax4.set_title('High Risk: p_loss >= 0.3', fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im_high, ax=ax4, label='Risk Level')
    
    # 5. Very high-risk pixels (threshold 0.5)
    ax5 = fig.add_subplot(gs[1, 1])
    very_high = np.where(np.isfinite(predictions), predictions >= 0.5, np.nan)
    im_vhigh = ax5.imshow(very_high, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax5.set_title('Very High Risk: p_loss >= 0.5', fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im_vhigh, ax=ax5, label='Risk Level')
    
    # 6. Statistics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    valid_pred = predictions[np.isfinite(predictions)]
    if len(valid_pred) > 0:
        stats_text = f"""
PREDICTION STATISTICS

Total predictions: {len(valid_pred):,}

Probability ranges:
  Min: {valid_pred.min():.3f}
  Max: {valid_pred.max():.3f}
  Mean: {valid_pred.mean():.3f}
  Median: {np.median(valid_pred):.3f}

Risk breakdown:
  Low risk (< 0.3): {np.sum(valid_pred < 0.3):,}
  High risk (>= 0.3): {np.sum(valid_pred >= 0.3):,}
  Very high (>= 0.5): {np.sum(valid_pred >= 0.5):,}

Predicted period: {year_t} → {year_tp1}
        """
    else:
        stats_text = "No valid predictions"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Deforestation Model Predictions: {year_t} → {year_tp1}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(out_dir / f'predictions_overlay_{year_t}_{year_tp1}.png', 
               dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved: predictions_overlay_{year_t}_{year_tp1}.png", flush=True)
    plt.close()


def save_prediction_map(predictions: np.ndarray, profile: dict, 
                       out_path: Path):
    """Save predictions as GeoTIFF for further analysis."""
    print(f"[VIZ] Saving prediction map to {out_path.name}...", flush=True)
    
    profile.update(
        count=1,
        dtype='float32',
        nodata=np.nan,
        compress='deflate',
    )
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(predictions, 1)
    
    print(f"[VIZ] Saved: {out_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Visualize deforestation predictions on satellite imagery')
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--year-t', type=int, required=True, help='Base year')
    parser.add_argument('--year-tp1', type=int, required=True, help='Target year')
    parser.add_argument('--model', type=Path, default=Path('model_outputs/deforestation_model.joblib'))
    parser.add_argument('--features', type=Path, default=Path('model_outputs/features.json'))
    parser.add_argument('--out-dir', type=Path, default=Path('model_outputs/visualizations'))
    parser.add_argument('--forest-ndvi-threshold', type=float, default=0.65)
    args = parser.parse_args()
    
    # Find and load satellite data
    raster_path = list(args.data_dir.glob(f'antioquia_{args.year_t}.tif'))
    if not raster_path:
        print(f"[VIZ] Error: No antioquia_{args.year_t}.tif found", flush=True)
        return
    raster_path = raster_path[0]
    
    # Load model and features
    model, features_list = load_model_and_features(args.model, args.features)
    
    # Make predictions
    predictions, ndvi_map, profile = predict_on_raster(
        raster_path, model, features_list,
        args.year_t, args.year_tp1,
        args.forest_ndvi_threshold
    )
    
    # Load satellite data for visualization
    with rasterio.open(raster_path) as src:
        satellite_data = src.read().astype(np.float32)
    
    # Visualize
    visualize_predictions(predictions, satellite_data, args.out_dir, 
                         args.year_t, args.year_tp1)
    
    # Save prediction map as GeoTIFF
    save_prediction_map(predictions, profile, 
                       args.out_dir / f'predictions_{args.year_t}_to_{args.year_tp1}.tif')
    
    print(f"\n[VIZ] Complete! Results saved to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
