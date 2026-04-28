from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import requests
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


YEARS = [2008, 2013, 2018, 2023]
OUT_DIR = Path("eda_antioquia_ndvi")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"
NDVI_ASSET = "250m_16_days_NDVI"
QUALITY_ASSET = "250m_16_days_pixel_reliability"
GEOB_API = "https://www.geoboundaries.org/api/current/gbOpen/COL/ADM1"


@dataclass
class AnnualNdvi:
    year: int
    ndvi: np.ndarray
    valid_count: np.ndarray
    transform: rasterio.Affine
    crs: str
    bounds_wgs84: tuple[float, float, float, float]


def get_antioquia_geometry():
    meta = requests.get(GEOB_API, timeout=60).json()
    gj = requests.get(meta["gjDownloadURL"], timeout=120).json()
    for feature in gj["features"]:
        name = feature["properties"].get("shapeName", "")
        if name.lower() == "antioquia":
            return shape(feature["geometry"]), feature
    raise RuntimeError("Antioquia boundary was not found in geoBoundaries ADM1.")


def stac_items_for_year(catalog: Client, aoi_geojson: dict, year: int):
    search = catalog.search(
        collections=[COLLECTION],
        intersects=aoi_geojson,
        datetime=f"{year}-01-01/{year}-12-31",
        limit=100,
    )
    items = list(search.items())
    items.sort(key=lambda item: item.properties.get("start_datetime", ""))
    return items


def projected_geometry(geom_wgs84, dst_crs):
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom_wgs84)


def yearly_mean_ndvi(year: int, items, geom_wgs84) -> AnnualNdvi:
    if not items:
        raise RuntimeError(f"No MODIS items found for {year}.")

    sample_href = planetary_computer.sign(items[0].assets[NDVI_ASSET].href)
    with rasterio.open(sample_href) as sample:
        geom_proj = projected_geometry(geom_wgs84, sample.crs)
        window = from_bounds(*geom_proj.bounds, transform=sample.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, sample.width, sample.height))
        out_transform = sample.window_transform(window)
        shape_hw = (int(window.height), int(window.width))
        inside = geometry_mask(
            [mapping(geom_proj)],
            out_shape=shape_hw,
            transform=out_transform,
            invert=True,
        )
        crs = sample.crs.to_string()

    ndvi_sum = np.zeros(shape_hw, dtype="float64")
    valid_count = np.zeros(shape_hw, dtype="uint16")

    for item in items:
        ndvi_href = planetary_computer.sign(item.assets[NDVI_ASSET].href)
        reliability_href = planetary_computer.sign(item.assets[QUALITY_ASSET].href)
        with rasterio.open(ndvi_href) as ndvi_ds:
            ndvi_raw = ndvi_ds.read(1, window=window).astype("float32")
        with rasterio.open(reliability_href) as rel_ds:
            reliability = rel_ds.read(1, window=window)

        valid = inside & np.isfinite(ndvi_raw)
        valid &= (ndvi_raw >= -2000) & (ndvi_raw <= 10000)
        valid &= (reliability == 0) | (reliability == 1)

        ndvi_sum[valid] += ndvi_raw[valid] * 0.0001
        valid_count[valid] += 1

    ndvi_mean = np.full(shape_hw, np.nan, dtype="float32")
    np.divide(ndvi_sum, valid_count, out=ndvi_mean, where=valid_count > 0)
    ndvi_mean[~inside] = np.nan

    return AnnualNdvi(
        year=year,
        ndvi=ndvi_mean,
        valid_count=valid_count,
        transform=out_transform,
        crs=crs,
        bounds_wgs84=geom_wgs84.bounds,
    )


def ndvi_palette():
    colors = [
        "#2d2d2d",
        "#8c510a",
        "#d8b365",
        "#f6e8c3",
        "#c7eae5",
        "#5ab4ac",
        "#01665e",
    ]
    bounds = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0]
    return ListedColormap(colors), BoundaryNorm(bounds, len(colors)), bounds


def save_map(annual: AnnualNdvi):
    cmap, norm, bounds = ndvi_palette()
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(annual.ndvi, cmap=cmap, norm=norm)
    ax.set_title(f"Antioquia - NDVI promedio anual {annual.year}")
    ax.set_axis_off()
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=bounds)
    cbar.set_label("NDVI")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ndvi_antioquia_{annual.year}.png", dpi=180)
    plt.close(fig)


def save_histogram(annual: AnnualNdvi):
    vals = annual.ndvi[np.isfinite(annual.ndvi)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=50, color="#2f7f6f", edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.nanmean(vals)), color="#b2182b", linewidth=2, label="media")
    ax.set_title(f"Distribucion NDVI - Antioquia {annual.year}")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("pixeles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"hist_ndvi_antioquia_{annual.year}.png", dpi=160)
    plt.close(fig)


def summarize_annual(annuals: list[AnnualNdvi]) -> pd.DataFrame:
    rows = []
    for item in annuals:
        vals = item.ndvi[np.isfinite(item.ndvi)]
        rows.append(
            {
                "year": item.year,
                "valid_pixels": int(vals.size),
                "mean_ndvi": float(np.nanmean(vals)),
                "median_ndvi": float(np.nanmedian(vals)),
                "std_ndvi": float(np.nanstd(vals)),
                "p10_ndvi": float(np.nanpercentile(vals, 10)),
                "p25_ndvi": float(np.nanpercentile(vals, 25)),
                "p75_ndvi": float(np.nanpercentile(vals, 75)),
                "p90_ndvi": float(np.nanpercentile(vals, 90)),
                "low_vegetation_pct": float(np.mean(vals < 0.4)),
                "medium_vegetation_pct": float(np.mean((vals >= 0.4) & (vals < 0.7))),
                "high_vegetation_pct": float(np.mean(vals >= 0.7)),
                "mean_valid_observations": float(np.mean(item.valid_count[item.valid_count > 0])),
            }
        )
    return pd.DataFrame(rows)


def build_grid_clusters(annuals: list[AnnualNdvi], block_size: int = 20, n_clusters: int = 5) -> pd.DataFrame:
    base = annuals[0]
    height, width = base.ndvi.shape
    inv_transformer = Transformer.from_crs(base.crs, "EPSG:4326", always_xy=True)
    rows = []

    for row0 in range(0, height, block_size):
        for col0 in range(0, width, block_size):
            row1 = min(row0 + block_size, height)
            col1 = min(col0 + block_size, width)
            features = {}
            valid_all_years = True
            for annual in annuals:
                block = annual.ndvi[row0:row1, col0:col1]
                if np.isfinite(block).sum() < 10:
                    valid_all_years = False
                    break
                features[f"ndvi_{annual.year}"] = float(np.nanmean(block))
            if not valid_all_years:
                continue

            center_col = (col0 + col1) / 2
            center_row = (row0 + row1) / 2
            x, y = base.transform * (center_col, center_row)
            lon, lat = inv_transformer.transform(x, y)
            rows.append(
                {
                    "grid_row": row0 // block_size,
                    "grid_col": col0 // block_size,
                    "lon": lon,
                    "lat": lat,
                    **features,
                }
            )

    df = pd.DataFrame(rows)
    feature_cols = ["lon", "lat"] + [f"ndvi_{annual.year}" for annual in annuals]
    x = StandardScaler().fit_transform(df[feature_cols])
    df["cluster"] = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(x)
    df["ndvi_change_2008_2023"] = df[f"ndvi_{YEARS[-1]}"] - df[f"ndvi_{YEARS[0]}"]
    return df


def save_cluster_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(df["lon"], df["lat"], c=df["cluster"], s=14, cmap="tab10", linewidths=0)
    ax.set_title("Clusters por ubicacion y NDVI medio por cuadricula")
    ax.set_xlabel("longitud")
    ax.set_ylabel("latitud")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clusters_antioquia_ndvi_grid.png", dpi=180)
    plt.close(fig)


def save_report(summary: pd.DataFrame, cluster_df: pd.DataFrame, item_counts: dict[int, int]):
    cluster_summary = (
        cluster_df.groupby("cluster")
        .agg(
            cells=("cluster", "size"),
            mean_lon=("lon", "mean"),
            mean_lat=("lat", "mean"),
            mean_ndvi_2008=("ndvi_2008", "mean"),
            mean_ndvi_2013=("ndvi_2013", "mean"),
            mean_ndvi_2018=("ndvi_2018", "mean"),
            mean_ndvi_2023=("ndvi_2023", "mean"),
            mean_change_2008_2023=("ndvi_change_2008_2023", "mean"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(OUT_DIR / "cluster_summary.csv", index=False)

    lines = [
        "# EDA NDVI Antioquia",
        "",
        "Producto: MODIS MOD13Q1/MYD13Q1 v6.1, NDVI 16 dias, 250 m.",
        "Metodo: promedio anual por pixel usando observaciones con confiabilidad buena o marginal.",
        "",
        "## Observaciones usadas",
        "",
        item_counts_to_markdown(item_counts),
        "",
        "## Resumen anual",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Clusters por cuadricula",
        "",
        cluster_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Archivos generados",
        "",
        "- `ndvi_antioquia_2008.png`, `ndvi_antioquia_2013.png`, `ndvi_antioquia_2018.png`, `ndvi_antioquia_2023.png`",
        "- `hist_ndvi_antioquia_2008.png`, `hist_ndvi_antioquia_2013.png`, `hist_ndvi_antioquia_2018.png`, `hist_ndvi_antioquia_2023.png`",
        "- `annual_ndvi_summary.csv`",
        "- `grid_clusters.csv`",
        "- `cluster_summary.csv`",
        "- `clusters_antioquia_ndvi_grid.png`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def item_counts_to_markdown(item_counts: dict[int, int]) -> str:
    df = pd.DataFrame({"year": list(item_counts), "modis_items": list(item_counts.values())})
    return df.to_markdown(index=False)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    geom_wgs84, antioquia_feature = get_antioquia_geometry()
    (OUT_DIR / "antioquia_boundary.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [antioquia_feature]}),
        encoding="utf-8",
    )

    catalog = Client.open(STAC_URL)
    aoi = mapping(geom_wgs84)
    annuals = []
    item_counts: dict[int, int] = {}

    for year in YEARS:
        items = stac_items_for_year(catalog, aoi, year)
        item_counts[year] = len(items)
        annual = yearly_mean_ndvi(year, items, geom_wgs84)
        annuals.append(annual)
        save_map(annual)
        save_histogram(annual)

    summary = summarize_annual(annuals)
    summary.to_csv(OUT_DIR / "annual_ndvi_summary.csv", index=False)

    cluster_df = build_grid_clusters(annuals)
    cluster_df.to_csv(OUT_DIR / "grid_clusters.csv", index=False)
    save_cluster_plot(cluster_df)
    save_report(summary, cluster_df, item_counts)

    print(summary.to_string(index=False))
    print(f"Outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
