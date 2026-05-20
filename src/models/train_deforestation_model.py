#!/usr/bin/env python
"""
Entrenamiento de un clasificador supervisado de deforestación T+1.

Flujo recomendado:
    1. Construir muestras de píxeles con build_supervised_dataset.py.
    2. Entrenar con validación temporal, usualmente el par más reciente disponible.

El modelo predice probabilidad de pérdida a nivel de píxel. Las estimaciones de área
se obtienen agregando probabilidades: sum(p_pérdida * área_píxel_ha).

Código realizado con apoyo de herramientas de inteligencia artificial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def make_model(random_state: int, n_estimators: int, learning_rate: float):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception:
        return HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=n_estimators,
            class_weight="balanced",
            random_state=random_state,
        )


def choose_features(df: pd.DataFrame, use_location: bool, use_year: bool) -> list[str]:
    excluded = {
        "target_loss",
        "pixel_area_ha",
        "year_tp1",
        "pair_index",
    }
    if not use_location:
        excluded.update({"row", "col", "lon", "lat"})
    if not use_year:
        excluded.update({"year_t", "prev_year"})

    features = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not features:
        raise ValueError("No numeric features available for training.")
    return features


def temporal_split(df: pd.DataFrame, holdout_year_t: int | None):
    available = sorted(df["year_t"].unique())
    if holdout_year_t is None:
        holdout_year_t = int(available[-1])
    if holdout_year_t not in available:
        raise ValueError(f"holdout_year_t={holdout_year_t} not in dataset years {available}")

    train_mask = df["year_t"] != holdout_year_t
    test_mask = df["year_t"] == holdout_year_t
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Temporal split produced empty train or test set.")
    return train_mask, test_mask, holdout_year_t


def threshold_table(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = proba >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def save_pr_curve(y_true: np.ndarray, proba: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_feature_importance(model, feature_names: list[str], out_path: Path) -> None:
    estimator = model.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out_path.with_suffix(".csv"), index=False)

    top = importance.head(25).iloc[::-1]
    plt.figure(figsize=(8, max(5, 0.3 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("model_outputs/train_pixels.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("model_outputs"))
    parser.add_argument("--holdout-year-t", type=int, default=None)
    parser.add_argument("--use-location", action="store_true")
    parser.add_argument("--use-year", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] Reading {args.dataset}", flush=True)
    df = pd.read_parquet(args.dataset)
    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)

    features = choose_features(df, args.use_location, args.use_year)
    train_mask, test_mask, holdout_year_t = temporal_split(df, args.holdout_year_t)

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, "target_loss"].astype(np.uint8)
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, "target_loss"].astype(np.uint8)

    preprocess = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                make_model(args.seed, args.n_estimators, args.learning_rate),
            ),
        ]
    )

    print(
        f"[TRAIN] rows train={len(X_train):,} test={len(X_test):,} holdout_year_t={holdout_year_t}",
        flush=True,
    )
    print(f"[TRAIN] features={len(features)}", flush=True)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred_05 = proba >= 0.5
    roc_auc = roc_auc_score(y_test, proba)
    avg_precision = average_precision_score(y_test, proba)

    metrics = {
        "dataset": str(args.dataset.resolve()),
        "holdout_year_t": int(holdout_year_t),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": features,
        "use_location": args.use_location,
        "use_year": args.use_year,
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
        "classification_report_threshold_0_5": classification_report(
            y_test, pred_05, output_dict=True, zero_division=0
        ),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    threshold_df = threshold_table(y_test.to_numpy(), proba)
    threshold_df.to_csv(args.out_dir / "threshold_metrics.csv", index=False)

    predictions = df.loc[test_mask, ["year_t", "year_tp1", "row", "col", "target_loss", "pixel_area_ha"]].copy()
    predictions["p_loss"] = proba.astype(np.float32)
    predictions.to_parquet(args.out_dir / "holdout_predictions.parquet", index=False)

    # On the sampled holdout set, this is not an unbiased area estimate unless
    # sampling weights are introduced. It is still useful for threshold behavior.
    sampled_area_summary = {
        "observed_loss_ha_in_sample": float(
            predictions.loc[predictions["target_loss"] == 1, "pixel_area_ha"].sum()
        ),
        "expected_loss_ha_in_sample": float(
            (predictions["p_loss"] * predictions["pixel_area_ha"]).sum()
        ),
    }
    (args.out_dir / "sampled_area_summary.json").write_text(
        json.dumps(sampled_area_summary, indent=2), encoding="utf-8"
    )

    joblib.dump(model, args.out_dir / "deforestation_model.joblib")
    (args.out_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    save_pr_curve(y_test.to_numpy(), proba, args.out_dir / "precision_recall_curve.png")
    save_feature_importance(model, features, args.out_dir / "feature_importance.png")

    print(f"[TRAIN] ROC AUC={roc_auc:.4f} AP={avg_precision:.4f}", flush=True)
    print(f"[TRAIN] Wrote outputs to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
