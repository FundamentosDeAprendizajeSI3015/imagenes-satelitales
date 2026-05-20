#!/usr/bin/env python
"""
Construcción de dataset supervisado para modelado de deforestación T+1.

Definición del target:
    y = 1 cuando un píxel parece bosque en T y se convierte en pérdida en T+1.
    y = 0 cuando un píxel parece bosque en T y permanece sin pérdida en T+1.

Crea un dataset clasificador a nivel de píxel. Las hectáreas se calculan luego
agregando predicciones: sum(probabilidad * área_píxel_ha).

Código realizado con apoyo de herramientas de inteligencia artificial.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import xy


BAND_NAMES = ["red", "nir", "swir1", "swir2", "ndvi"]
PIXEL_AREA_HA_60M = 0.36


def find_rasters(data_dir: Path) -> dict[int, Path]:
    rasters = {}
    for path in data_dir.glob("antioquia_*.tif"):
        match = re.fullmatch(r"antioquia_(\d{4})\.tif", path.name, flags=re.IGNORECASE)
        if match:
            rasters[int(match.group(1))] = path
    if len(rasters) < 2:
        raise SystemExit(f"Need at least two antioquia_*.tif files in {data_dir}")
    return dict(sorted(rasters.items()))


def rasters_aligned(
    src_a: rasterio.DatasetReader,
    src_b: rasterio.DatasetReader,
    transform_tolerance: float,
) -> bool:
    same_shape = (src_a.width, src_a.height) == (src_b.width, src_b.height)
    same_crs = src_a.crs == src_b.crs
    same_transform = np.allclose(
        tuple(src_a.transform),
        tuple(src_b.transform),
        rtol=0,
        atol=transform_tolerance,
    )
    return bool(same_shape and same_crs and same_transform)


def alignment_report(src_a: rasterio.DatasetReader, src_b: rasterio.DatasetReader) -> dict:
    return {
        "a_file": Path(src_a.name).name,
        "b_file": Path(src_b.name).name,
        "a_shape": [src_a.height, src_a.width],
        "b_shape": [src_b.height, src_b.width],
        "a_crs": str(src_a.crs),
        "b_crs": str(src_b.crs),
        "a_transform": tuple(src_a.transform),
        "b_transform": tuple(src_b.transform),
        "a_bounds": tuple(src_a.bounds),
        "b_bounds": tuple(src_b.bounds),
    }


def safe_index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def make_features(
    data_t: np.ndarray,
    rows_abs: np.ndarray,
    cols_abs: np.ndarray,
    year_t: int,
    year_tp1: int,
    pair_index: int,
    src_t: rasterio.DatasetReader,
    include_coords: bool,
    data_prev: np.ndarray | None = None,
    prev_year: int | None = None,
) -> dict[str, np.ndarray]:
    red, nir, swir1, swir2, ndvi = data_t
    ndmi = safe_index(nir - swir1, nir + swir1)
    nbr = safe_index(nir - swir2, nir + swir2)
    swir_ratio = safe_index(swir1, swir2)

    rr = rows_abs.astype(np.int64)
    cc = cols_abs.astype(np.int64)
    local_rows = rr - int(rows_abs.min())
    local_cols = cc - int(cols_abs.min())

    # local_rows/local_cols are overwritten below by caller-sliced arrays via flat indices.
    # This function receives full absolute rows/cols and feature arrays already flattened.
    features = {
        "year_t": np.full(rr.shape, year_t, dtype=np.int16),
        "year_tp1": np.full(rr.shape, year_tp1, dtype=np.int16),
        "interval_years": np.full(rr.shape, year_tp1 - year_t, dtype=np.int16),
        "pair_index": np.full(rr.shape, pair_index, dtype=np.int16),
        "row": rr.astype(np.int32),
        "col": cc.astype(np.int32),
        "red_t": red,
        "nir_t": nir,
        "swir1_t": swir1,
        "swir2_t": swir2,
        "ndvi_t": ndvi,
        "ndmi_t": ndmi,
        "nbr_t": nbr,
        "swir1_swir2_ratio_t": swir_ratio,
    }

    if include_coords:
        lon, lat = xy(src_t.transform, rr, cc, offset="center")
        features["lon"] = np.asarray(lon, dtype=np.float32)
        features["lat"] = np.asarray(lat, dtype=np.float32)

    if data_prev is not None and prev_year is not None:
        prev_ndvi = data_prev[4]
        features["prev_year"] = np.full(rr.shape, prev_year, dtype=np.int16)
        features["prev_interval_years"] = np.full(rr.shape, year_t - prev_year, dtype=np.int16)
        features["ndvi_prev"] = prev_ndvi
        features["delta_ndvi_prev_to_t"] = ndvi - prev_ndvi
        features["nbr_prev"] = safe_index(data_prev[1] - data_prev[3], data_prev[1] + data_prev[3])
        features["delta_nbr_prev_to_t"] = features["nbr_t"] - features["nbr_prev"]
    else:
        features["prev_year"] = np.full(rr.shape, -1, dtype=np.int16)
        features["prev_interval_years"] = np.full(rr.shape, -1, dtype=np.int16)
        features["ndvi_prev"] = np.full(rr.shape, np.nan, dtype=np.float32)
        features["delta_ndvi_prev_to_t"] = np.full(rr.shape, np.nan, dtype=np.float32)
        features["nbr_prev"] = np.full(rr.shape, np.nan, dtype=np.float32)
        features["delta_nbr_prev_to_t"] = np.full(rr.shape, np.nan, dtype=np.float32)

    return features


def flatten_selected(arrays: dict[str, np.ndarray], flat_idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value.reshape(-1)[flat_idx] for key, value in arrays.items()}


def sample_indices(
    positive_flat: np.ndarray,
    negative_flat: np.ndarray,
    rng: np.random.Generator,
    negative_ratio: float,
    max_positive: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if positive_flat.size == 0:
        return positive_flat, positive_flat

    pos = positive_flat
    if max_positive is not None and pos.size > max_positive:
        pos = rng.choice(pos, size=max_positive, replace=False)

    n_neg = min(negative_flat.size, int(math.ceil(pos.size * negative_ratio)))
    if n_neg == 0:
        return pos, np.array([], dtype=np.int64)

    neg = rng.choice(negative_flat, size=n_neg, replace=False)
    return pos, neg


def flush_buffers(
    buffers: list[pd.DataFrame],
    out_path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    if not buffers:
        return writer
    df = pd.concat(buffers, ignore_index=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
    writer.write_table(table)
    buffers.clear()
    return writer


def selected_rows_cols(window, flat_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_rows, local_cols = np.unravel_index(flat_idx, (window.height, window.width))
    rows_abs = local_rows.astype(np.int64) + window.row_off
    cols_abs = local_cols.astype(np.int64) + window.col_off
    return rows_abs, cols_abs


def iter_pairs(years: list[int]) -> Iterable[tuple[int | None, int, int]]:
    for idx, year_t in enumerate(years[:-1]):
        prev_year = years[idx - 1] if idx > 0 else None
        yield prev_year, year_t, years[idx + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("model_outputs/train_pixels.parquet"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--forest-ndvi-threshold", type=float, default=0.65)
    parser.add_argument("--loss-ndvi-threshold", type=float, default=0.45)
    parser.add_argument("--min-ndvi-drop", type=float, default=-0.20)
    parser.add_argument("--max-positive-per-window", type=int, default=3000)
    parser.add_argument("--flush-rows", type=int, default=250_000)
    parser.add_argument("--include-coords", action="store_true")
    parser.add_argument(
        "--transform-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for tiny floating point differences in raster transforms.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rasters = find_rasters(args.data_dir)
    years = list(rasters)
    print(
        "[DATASET] Using rasters: "
        + ", ".join(f"{year}:{path.name}" for year, path in rasters.items()),
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.exists():
        args.out.unlink()

    writer = None
    buffers: list[pd.DataFrame] = []
    manifest_rows = []

    for pair_index, (prev_year, year_t, year_tp1) in enumerate(iter_pairs(years)):
        print(f"[DATASET] Pair {year_t} -> {year_tp1}", flush=True)
        pair_rows = 0
        pair_pos = 0
        pair_neg = 0

        with rasterio.open(rasters[year_t]) as src_t, rasterio.open(rasters[year_tp1]) as src_tp1:
            src_prev = rasterio.open(rasters[prev_year]) if prev_year is not None else None
            try:
                if not rasters_aligned(src_t, src_tp1, args.transform_tolerance):
                    raise ValueError(
                        f"Rasters not aligned for {year_t}->{year_tp1}: "
                        f"{json.dumps(alignment_report(src_t, src_tp1), indent=2)}"
                    )

                for _, window in src_t.block_windows(1):
                    data_t = src_t.read(window=window, masked=False).astype(np.float32)
                    data_tp1 = src_tp1.read(window=window, masked=False).astype(np.float32)
                    data_prev = (
                        src_prev.read(window=window, masked=False).astype(np.float32)
                        if src_prev is not None
                        else None
                    )

                    valid_features = np.all(np.isfinite(data_t), axis=0)
                    valid_target = np.isfinite(data_t[4]) & np.isfinite(data_tp1[4])
                    forest_t = data_t[4] >= args.forest_ndvi_threshold
                    loss = (
                        valid_features
                        & valid_target
                        & forest_t
                        & (data_tp1[4] <= args.loss_ndvi_threshold)
                        & ((data_tp1[4] - data_t[4]) <= args.min_ndvi_drop)
                    )
                    stable_forest = valid_features & valid_target & forest_t & ~loss

                    pos_flat, neg_flat = sample_indices(
                        np.flatnonzero(loss),
                        np.flatnonzero(stable_forest),
                        rng,
                        args.negative_ratio,
                        args.max_positive_per_window,
                    )
                    if pos_flat.size == 0 or neg_flat.size == 0:
                        continue

                    selected = np.concatenate([pos_flat, neg_flat])
                    y = np.concatenate(
                        [
                            np.ones(pos_flat.size, dtype=np.uint8),
                            np.zeros(neg_flat.size, dtype=np.uint8),
                        ]
                    )
                    order = rng.permutation(selected.size)
                    selected = selected[order]
                    y = y[order]

                    rows_abs, cols_abs = selected_rows_cols(window, selected)
                    local_arrays = {
                        name: data_t[idx] for idx, name in enumerate(BAND_NAMES)
                    }
                    if data_prev is not None:
                        prev_arrays = {
                            name: data_prev[idx] for idx, name in enumerate(BAND_NAMES)
                        }
                        prev_selected = flatten_selected(prev_arrays, selected)
                        data_prev_selected = np.vstack(
                            [prev_selected[name] for name in BAND_NAMES]
                        ).astype(np.float32)
                    else:
                        data_prev_selected = None

                    selected_features = flatten_selected(local_arrays, selected)
                    data_t_selected = np.vstack(
                        [selected_features[name] for name in BAND_NAMES]
                    ).astype(np.float32)

                    feature_dict = make_features(
                        data_t_selected,
                        rows_abs,
                        cols_abs,
                        year_t,
                        year_tp1,
                        pair_index,
                        src_t,
                        args.include_coords,
                        data_prev_selected,
                        prev_year,
                    )
                    feature_dict["target_loss"] = y
                    feature_dict["pixel_area_ha"] = np.full(
                        y.shape, PIXEL_AREA_HA_60M, dtype=np.float32
                    )
                    df = pd.DataFrame(feature_dict)
                    buffers.append(df)

                    pair_rows += len(df)
                    pair_pos += int(y.sum())
                    pair_neg += int((y == 0).sum())

                    if sum(len(chunk) for chunk in buffers) >= args.flush_rows:
                        writer = flush_buffers(buffers, args.out, writer)
            finally:
                if src_prev is not None:
                    src_prev.close()

        manifest_rows.append(
            {
                "year_t": year_t,
                "year_tp1": year_tp1,
                "rows": pair_rows,
                "positives": pair_pos,
                "negatives": pair_neg,
                "positive_rate": pair_pos / pair_rows if pair_rows else 0.0,
            }
        )
        print(
            f"[DATASET] rows={pair_rows:,} positives={pair_pos:,} negatives={pair_neg:,}",
            flush=True,
        )

    writer = flush_buffers(buffers, args.out, writer)
    if writer is not None:
        writer.close()

    manifest = {
        "dataset": str(args.out.resolve()),
        "years": years,
        "negative_ratio": args.negative_ratio,
        "forest_ndvi_threshold": args.forest_ndvi_threshold,
        "loss_ndvi_threshold": args.loss_ndvi_threshold,
        "min_ndvi_drop": args.min_ndvi_drop,
        "include_coords": args.include_coords,
        "pixel_area_ha": PIXEL_AREA_HA_60M,
        "pairs": manifest_rows,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DATASET] Wrote {args.out}", flush=True)
    print(f"[DATASET] Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
