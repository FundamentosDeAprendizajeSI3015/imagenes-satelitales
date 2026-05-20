#!/usr/bin/env python
"""
Aplicar un modelo de deforestación T+1 entrenado a un raster completo.

La salida es un GeoTIFF de una banda con probabilidad de pérdida a nivel de píxel
para píxeles que parecen bosque en T. Los píxeles que no son bosque o inválidos
se escriben como NaN.

Código realizado con apoyo de herramientas de inteligencia artificial.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.data.build_supervised_dataset import BAND_NAMES, make_features, selected_rows_cols


def find_rasters(data_dir: Path) -> dict[int, Path]:
    rasters = {}
    for path in data_dir.glob("antioquia_*.tif"):
        match = re.fullmatch(r"antioquia_(\d{4})\.tif", path.name, flags=re.IGNORECASE)
        if match:
            rasters[int(match.group(1))] = path
    return dict(sorted(rasters.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--year-t", type=int, required=True)
    parser.add_argument("--prev-year", type=int, default=None)
    parser.add_argument("--interval-years", type=int, default=1)
    parser.add_argument("--model", type=Path, default=Path("model_outputs/deforestation_model.joblib"))
    parser.add_argument("--features", type=Path, default=Path("model_outputs/features.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--forest-ndvi-threshold", type=float, default=0.65)
    args = parser.parse_args()

    rasters = find_rasters(args.data_dir)
    if args.year_t not in rasters:
        raise SystemExit(f"No raster found for year_t={args.year_t}")
    if args.prev_year is not None and args.prev_year not in rasters:
        raise SystemExit(f"No raster found for prev_year={args.prev_year}")

    out_path = args.out or Path(f"model_outputs/p_loss_{args.year_t}_to_tplus{args.interval_years}.tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    features = json.loads(args.features.read_text(encoding="utf-8"))

    with rasterio.open(rasters[args.year_t]) as src_t:
        src_prev = rasterio.open(rasters[args.prev_year]) if args.prev_year is not None else None
        try:
            profile = src_t.profile.copy()
            profile.update(
                count=1,
                dtype="float32",
                nodata=np.nan,
                compress="deflate",
                predictor=3,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, window in src_t.block_windows(1):
                    data_t = src_t.read(window=window, masked=False).astype(np.float32)
                    valid = np.all(np.isfinite(data_t), axis=0) & (
                        data_t[4] >= args.forest_ndvi_threshold
                    )
                    out = np.full((window.height, window.width), np.nan, dtype=np.float32)

                    if valid.any():
                        flat_idx = np.flatnonzero(valid)
                        rows_abs, cols_abs = selected_rows_cols(window, flat_idx)
                        local_arrays = {
                            name: data_t[idx] for idx, name in enumerate(BAND_NAMES)
                        }
                        selected_features = {
                            key: value.reshape(-1)[flat_idx]
                            for key, value in local_arrays.items()
                        }
                        data_t_selected = np.vstack(
                            [selected_features[name] for name in BAND_NAMES]
                        ).astype(np.float32)

                        if src_prev is not None:
                            data_prev = src_prev.read(window=window, masked=False).astype(np.float32)
                            prev_selected = np.vstack(
                                [
                                    data_prev[idx].reshape(-1)[flat_idx]
                                    for idx, _ in enumerate(BAND_NAMES)
                                ]
                            ).astype(np.float32)
                        else:
                            prev_selected = None

                        feature_dict = make_features(
                            data_t_selected,
                            rows_abs,
                            cols_abs,
                            args.year_t,
                            args.year_t + args.interval_years,
                            pair_index=-1,
                            src_t=src_t,
                            include_coords=any(col in features for col in ["lon", "lat"]),
                            data_prev=prev_selected,
                            prev_year=args.prev_year,
                        )
                        df = pd.DataFrame(feature_dict)
                        missing = [col for col in features if col not in df.columns]
                        if missing:
                            raise ValueError(f"Missing model features at prediction time: {missing}")
                        proba = model.predict_proba(df[features])[:, 1].astype(np.float32)
                        out.reshape(-1)[flat_idx] = proba

                    dst.write(out, 1, window=window)
        finally:
            if src_prev is not None:
                src_prev.close()

    print(f"[PREDICT] Wrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
