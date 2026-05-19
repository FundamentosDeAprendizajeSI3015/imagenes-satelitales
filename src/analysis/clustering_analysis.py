#!/usr/bin/env python3
"""Unified clustering analysis: PCA + K-means + spatial comparison."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

BAND_NAMES = ["red", "nir", "swir1", "swir2", "ndvi"]


def safe_index(numerator, denominator):
    """Compute spectral index safely."""
    return np.divide(numerator, denominator, 
                     where=denominator!=0, 
                     out=np.zeros_like(numerator, dtype=float))


def compute_ndvi(red, nir):
    return safe_index(nir - red, nir + red)


def load_raster(raster_path: Path, sample_size: int = 50000):
    """Load and sample raster data."""
    print(f"[CLUSTER] Loading {raster_path.name}...", flush=True)
    with rasterio.open(raster_path) as src:
        data = src.read().astype(np.float32)
    
    H, W = data.shape[1], data.shape[2]
    data_flat = data.reshape(5, -1).T
    
    valid_mask = np.all(np.isfinite(data_flat), axis=1)
    data_valid = data_flat[valid_mask]
    
    print(f"[CLUSTER] Valid pixels: {len(data_valid):,} / {len(data_flat):,}", flush=True)
    
    if len(data_valid) > sample_size:
        idx = np.random.choice(len(data_valid), size=sample_size, replace=False)
        data_valid = data_valid[idx]
        print(f"[CLUSTER] Sampled to {sample_size:,} pixels", flush=True)
    
    return data_valid, data, H, W


def pca_analysis(X: np.ndarray, n_components: int = 2):
    """Perform PCA."""
    print(f"[CLUSTER] Performing PCA...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    print(f"[CLUSTER] PCA variance explained: {np.cumsum(explained_var)[-1]:.2%}", flush=True)
    return X_pca, explained_var, pca, scaler


def kmeans_analysis(X: np.ndarray, k_range: range = range(2, 9)):
    """Test K-means for different k values."""
    print(f"[CLUSTER] Testing K-means...", flush=True)
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=0)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        results.append({
            'k': k,
            'silhouette': sil,
            'davies_bouldin': db,
            'inertia': kmeans.inertia_,
            'labels': labels
        })
        print(f"[CLUSTER]   k={k}: Silhouette={sil:.3f}, DB={db:.3f}", flush=True)
    return results


def save_pca_plot(X_pca, labels, explained_var, out_dir):
    """Save PCA scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', 
                        alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    ax.set_title('PCA: Pixel Clusters in 2D Space')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    out_path = out_dir / 'pca_clusters_2d.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CLUSTER] Saved: {out_path.name}", flush=True)


def save_kmeans_plots(results, out_dir):
    """Save K-means metrics plots."""
    k_values = [r['k'] for r in results]
    silhouette_scores = [r['silhouette'] for r in results]
    davies_bouldin_scores = [r['davies_bouldin'] for r in results]
    inertias = [r['inertia'] for r in results]
    
    optimal_k_sil = k_values[np.argmax(silhouette_scores)]
    optimal_k_db = k_values[np.argmin(davies_bouldin_scores)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(k_values, silhouette_scores, 'o-', linewidth=2, markersize=8, color='green')
    axes[0].axvline(optimal_k_sil, color='red', linestyle='--', label=f'Optimal k={optimal_k_sil}')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score (higher is better)')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(k_values, davies_bouldin_scores, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].axvline(optimal_k_db, color='red', linestyle='--', label=f'Optimal k={optimal_k_db}')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Davies-Bouldin Index')
    axes[1].set_title('Davies-Bouldin Index (lower is better)')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(k_values, inertias, 'o-', linewidth=2, markersize=8, color='purple')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Within-Cluster Sum of Squares')
    axes[2].set_title('Elbow Method')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / 'kmeans_metrics.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CLUSTER] Saved: {out_path.name}", flush=True)
    
    metrics_df = pd.DataFrame({
        'k': k_values,
        'silhouette_score': silhouette_scores,
        'davies_bouldin_index': davies_bouldin_scores,
        'inertia': inertias,
    })
    csv_path = out_dir / 'kmeans_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"[CLUSTER] Saved: {csv_path.name}", flush=True)
    
    return optimal_k_sil, optimal_k_db


def load_chunk(raster_path, row_start, col_start, chunk_size=512):
    """Load a 512x512 chunk."""
    with rasterio.open(raster_path) as src:
        row_end = min(row_start + chunk_size, src.height)
        col_end = min(col_start + chunk_size, src.width)
        window = rasterio.windows.Window(col_start, row_start, 
                                         col_end - col_start, row_end - row_start)
        data = src.read(window=window).astype(np.float32)
        
        if data.shape[1] < chunk_size or data.shape[2] < chunk_size:
            padded = np.full((data.shape[0], chunk_size, chunk_size), np.nan)
            padded[:, :data.shape[1], :data.shape[2]] = data
            data = padded
    return data


def process_chunk(red, nir, swir1, swir2):
    """Cluster a chunk."""
    ndvi = compute_ndvi(red, nir)
    X = np.stack([red, nir, swir1, swir2], axis=-1)
    h, w = X.shape[:2]
    X_flat = X.reshape(-1, 4)
    
    valid_mask = np.all(np.isfinite(X_flat), axis=1)
    X_valid = X_flat[valid_mask]
    
    if len(X_valid) == 0:
        return ndvi, np.zeros((h, w), dtype=np.int8), {}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    cluster_map = np.zeros((h, w), dtype=np.int8)
    cluster_map[~valid_mask.reshape(h, w)] = -1
    cluster_map[valid_mask.reshape(h, w)] = labels
    
    stats = {
        'ndvi_mean': float(np.nanmean(ndvi)),
        'ndvi_std': float(np.nanstd(ndvi)),
        'valid_pixels': int(np.sum(valid_mask)),
        'pca_variance': float(pca.explained_variance_ratio_.sum()),
        'cluster_counts': {str(i): int(np.sum(labels == i)) for i in range(3)}
    }
    return ndvi, cluster_map, stats


def select_chunks(raster_path, num_chunks=6, chunk_size=512):
    """Select strategic chunks."""
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
    
    n_rows = height // chunk_size
    n_cols = width // chunk_size
    
    chunks = []
    corners = [(0, 0), (0, (n_cols-1)*chunk_size), 
               ((n_rows-1)*chunk_size, 0), ((n_rows-1)*chunk_size, (n_cols-1)*chunk_size)]
    
    for i, (r, c) in enumerate(corners[:min(4, num_chunks)]):
        chunks.append((i+1, r, c))
    
    if num_chunks > 4:
        chunks.append((5, (n_rows//2)*chunk_size, (n_cols//2)*chunk_size))
    
    return chunks


def save_comparison_figure(year, chunks_data, out_dir):
    """Save NDVI vs Clustering comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = len(chunks_data)
    fig = plt.figure(figsize=(16, 3*n_chunks))
    gs = GridSpec(n_chunks, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    all_stats = []
    for idx, (chunk_id, ndvi, cluster_map, stats) in enumerate(chunks_data):
        # NDVI
        ax_ndvi = fig.add_subplot(gs[idx, 0])
        ndvi_valid = np.where(np.isfinite(ndvi), ndvi, np.nan)
        im_ndvi = ax_ndvi.imshow(ndvi_valid, cmap='RdYlGn', vmin=0, vmax=1)
        ax_ndvi.set_title(f'Chunk {chunk_id}: NDVI (μ={stats["ndvi_mean"]:.2f})')
        ax_ndvi.axis('off')
        plt.colorbar(im_ndvi, ax=ax_ndvi, label='NDVI')
        
        # Clustering
        ax_cluster = fig.add_subplot(gs[idx, 1])
        cluster_display = np.where(cluster_map == -1, np.nan, cluster_map)
        colors_cluster = ['#2ecc71', '#3498db', '#e74c3c']
        cmap_cluster = plt.cm.colors.ListedColormap(colors_cluster)
        im_cluster = ax_cluster.imshow(cluster_display, cmap=cmap_cluster, vmin=0, vmax=2)
        ax_cluster.set_title(f'Chunk {chunk_id}: Clustering (k=3)')
        ax_cluster.axis('off')
        handles = [mpatches.Patch(facecolor=colors_cluster[i], label=f'Cluster {i}') 
                   for i in range(3)]
        ax_cluster.legend(handles=handles, loc='upper right', fontsize=8)
        
        # Stats
        ax_stats = fig.add_subplot(gs[idx, 2])
        ax_stats.axis('off')
        stats_text = (
            f"Chunk {chunk_id}\nYear: {year}\n"
            f"━━━━━━━━━━━━━\n"
            f"NDVI Mean: {stats['ndvi_mean']:.3f}\n"
            f"NDVI Std: {stats['ndvi_std']:.3f}\n"
            f"Valid: {stats['valid_pixels']:,}\n"
            f"PCA Var: {stats['pca_variance']:.1%}\n"
            f"━━━━━━━━━━━━━\n"
            f"Cluster 0: {stats['cluster_counts']['0']:,}\n"
            f"Cluster 1: {stats['cluster_counts']['1']:,}\n"
            f"Cluster 2: {stats['cluster_counts']['2']:,}\n"
        )
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        all_stats.append({**stats, 'chunk_id': chunk_id})
    
    fig.suptitle(f'NDVI vs Clustering Comparison - Year {year}\n(Strategic 512×512 Chunks)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    out_path = out_dir / f'ndvi_clustering_comparison_{year}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[CLUSTER] Saved: {out_path.name}", flush=True)
    
    stats_path = out_dir / f'chunk_comparison_stats_{year}.json'
    with open(stats_path, 'w') as f:
        json.dump({"year": year, "chunks": all_stats}, f, indent=2)
    print(f"[CLUSTER] Saved: {stats_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Unified clustering analysis')
    parser.add_argument('--data-dir', type=Path, default=Path('raw_data'))
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/clustering'))
    parser.add_argument('--sample-size', type=int, default=50000)
    parser.add_argument('--num-chunks', type=int, default=6)
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find raster
    raster_path = list(args.data_dir.glob(f'antioquia_{args.year}.tif'))
    if not raster_path:
        print(f"[ERROR] No raster found for year {args.year}")
        return 1
    raster_path = raster_path[0]
    
    # 1. PCA + K-means analysis on sampled data
    X, full_data, H, W = load_raster(raster_path, sample_size=args.sample_size)
    X_pca, explained_var, pca, scaler = pca_analysis(X)
    results = kmeans_analysis(X_pca)
    
    # Save plots
    optimal_k = results[1]['labels']  # k=3
    save_pca_plot(X_pca, optimal_k, explained_var, args.out_dir)
    save_kmeans_plots(results, args.out_dir)
    
    # 2. Chunk-based comparison
    print(f"\n[CLUSTER] Processing strategic chunks...", flush=True)
    chunks = select_chunks(raster_path, num_chunks=args.num_chunks)
    chunks_data = []
    for chunk_id, row_start, col_start in chunks:
        data = load_chunk(raster_path, row_start, col_start)
        red, nir, swir1, swir2 = data[0], data[1], data[2], data[3]
        ndvi, cluster_map, stats = process_chunk(red, nir, swir1, swir2)
        chunks_data.append((chunk_id, ndvi, cluster_map, stats))
        print(f"[CLUSTER]   Chunk {chunk_id}: done", flush=True)
    
    save_comparison_figure(args.year, chunks_data, args.out_dir)
    
    print(f"\n[CLUSTER] ✓ Analysis complete!")
    print(f"[CLUSTER] Results saved to {args.out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    exit(main())
