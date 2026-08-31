"""Retrain an emulator from any conforming dataset, tracked in MLflow.

Takes a dataset config (see ``training/dataset_configs/``), runs the AutoEmulate
comparison, logs params / metrics / artifacts to MLflow, and — **only if the new model
improves on the currently registered best** — writes a new versioned directory into the
registry.

Two rules this module will not break:

* It never overwrites an existing version. A new model is always a new version
  directory; history is append-only.
* It never promotes automatically in CI. ``--promote`` writes the version; without it
  the run is a dry run that reports what it *would* do. CI runs the dry form and posts
  a report — a human merges.

Usage:
    python training/retrain.py --dataset-config training/dataset_configs/battery_capacity.yaml
    python training/retrain.py --dataset-config ... --promote
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "registry"
sys.path.insert(0, str(REPO))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrain an emulator and report on promotion.")
    ap.add_argument("--dataset-config", required=True, help="Path to a dataset config YAML")
    ap.add_argument("--promote", action="store_true",
                    help="Write the new version to the registry if it improves")
    ap.add_argument("--report", default=None, help="Write a markdown report to this path")
    ap.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    ap.add_argument("--quick", action="store_true", help="Tiny search budget, for CI smoke runs")
    return ap.parse_args(argv)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_value(metrics: dict, name: str) -> float:
    """AutoEmulate metrics are keyed by Metric objects with (mean, std) tuple values."""
    for key, val in metrics.items():
        if getattr(key, "name", str(key)) == name:
            return float(val[0] if isinstance(val, tuple) else val)
    return float("nan")


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, Path]:
    path = Path(cfg["dataset"])
    if not path.is_absolute():
        path = (REPO / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    needed = list(cfg["inputs"]) + [cfg["output"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset {path} is missing columns: {missing}")
    return df.dropna(subset=needed).reset_index(drop=True), path


def current_best(model_id: str) -> tuple[str | None, float | None]:
    """Highest registered version of ``model_id`` and its recorded held-out R2."""
    versions = []
    for manifest_path in (REGISTRY / model_id).glob("*/manifest.json") \
            if (REGISTRY / model_id).is_dir() else []:
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            versions.append((tuple(int(p) for p in m["version"].split(".")), m))
        except Exception:  # noqa: BLE001 - a broken manifest simply does not count
            continue
    if not versions:
        return None, None
    _, best = max(versions, key=lambda t: t[0])
    return best["version"], float(best["metrics"]["r2"])


def bump_version(version: str | None) -> str:
    if version is None:
        return "1.0.0"
    major, minor, patch = (int(p) for p in version.split("."))
    return f"{major}.{minor + 1}.0"


def train(cfg: dict, df: pd.DataFrame, quick: bool):
    from autoemulate import AutoEmulate

    inputs, output = list(cfg["inputs"]), cfg["output"]
    x = df[inputs].to_numpy(dtype=np.float64)
    y = df[[output]].to_numpy(dtype=np.float64)
    seed = int(cfg.get("seed", 42))
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=float(cfg.get("test_size", 0.2)), random_state=seed
    )
    ae = AutoEmulate(
        x_tr, y_tr,
        test_data=(x_te, y_te),
        models=cfg.get("models", ["GaussianProcessRBF", "GaussianProcessMatern32", "MLP"]),
        n_iter=2 if quick else int(cfg.get("n_iter", 10)),
        n_splits=2 if quick else int(cfg.get("n_splits", 3)),
        random_seed=seed,
        log_level="error",
    )
    return ae, (x_tr, x_te, y_tr, y_te)


def write_version(cfg: dict, ae, best, version: str, df: pd.DataFrame,
                  dataset_path: Path, n_train: int, n_test: int, r2: float,
                  rmse: float) -> Path:
    """Write a new version directory. Refuses to touch an existing one."""
    model_id = cfg["model_id"]
    target = REGISTRY / model_id / version
    if target.exists():
        raise FileExistsError(
            f"{target} already exists. Versions are immutable; bump the version instead."
        )
    target.mkdir(parents=True, exist_ok=True)
    ae.save(best, target / "emulator", use_timestamp=False)

    manifest = {
        "model_id": model_id,
        "version": version,
        "project": cfg.get("project", "retrained"),
        "description": cfg.get("description"),
        "autoemulate_version": pkg_version("autoemulate"),
        "emulator_model": best.model_name,
        "training_date": dt.datetime.now(dt.UTC).date().isoformat(),
        "dataset_hash": sha256_file(dataset_path),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "inputs": [
            {
                "name": name,
                "unit": cfg.get("units", {}).get(name, "-"),
                "min": float(df[name].min()),
                "max": float(df[name].max()),
            }
            for name in cfg["inputs"]
        ],
        "output": {
            "name": cfg["output"],
            "unit": cfg.get("units", {}).get(cfg["output"], "-"),
        },
        "metrics": {"r2": round(r2, 6), "rmse": round(rmse, 6)},
        "artifact": "emulator",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def markdown_report(cfg: dict, old_version, old_r2, new_version, r2, rmse,
                    model_name, improved, promoted, n_train, n_test) -> str:
    delta = "n/a" if old_r2 is None else f"{r2 - old_r2:+.4f}"
    recommendation = (
        "**PROMOTE** -- the retrained model improves held-out R2."
        if improved
        else "**REJECT** -- the retrained model does not improve on the registered version."
    )
    return "\n".join([
        f"## Retraining report: `{cfg['model_id']}`",
        "",
        f"Output: `{cfg['output']}` | Winning model: `{model_name}` "
        f"| Train/test: {n_train}/{n_test}",
        "",
        "| | Registered | Retrained | Delta |",
        "|---|---|---|---|",
        f"| Version | {old_version or '_none_'} | {new_version} | |",
        f"| R2 (held out) | {'n/a' if old_r2 is None else f'{old_r2:.4f}'} | {r2:.4f} | {delta} |",
        f"| RMSE | n/a | {rmse:.4g} | |",
        "",
        f"### Recommendation: {recommendation}",
        "",
        (
            f"A new version directory `registry/{cfg['model_id']}/{new_version}/` was written."
            if promoted
            else "_No registry change was made. Automation proposes; a human merges._"
        ),
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))

    df, dataset_path = load_dataset(cfg)
    print(f"Dataset {dataset_path}: {len(df)} usable rows")

    ae, (x_tr, x_te, _, _) = train(cfg, df, args.quick)
    best = ae.best_result()
    r2 = metric_value(best.test_metrics, "r2")
    rmse = metric_value(best.test_metrics, "rmse")
    print(f"Best model: {best.model_name}  R2={r2:.4f}  RMSE={rmse:.4g}")

    old_version, old_r2 = current_best(cfg["model_id"])
    new_version = bump_version(old_version)
    improved = old_r2 is None or r2 > old_r2

    # ---- MLflow tracking -----------------------------------------------------
    try:
        import mlflow

        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(f"emulator-retrain/{cfg['model_id']}")
        with mlflow.start_run(run_name=f"{cfg['model_id']}-{new_version}"):
            mlflow.log_params({
                "model_id": cfg["model_id"],
                "output": cfg["output"],
                "n_inputs": len(cfg["inputs"]),
                "n_train": len(x_tr),
                "n_test": len(x_te),
                "candidate_version": new_version,
                "winning_model": best.model_name,
                "autoemulate_version": pkg_version("autoemulate"),
                "dataset_hash": sha256_file(dataset_path),
            })
            mlflow.log_metrics({"r2_test": r2, "rmse_test": rmse})
            if old_r2 is not None:
                mlflow.log_metrics({"r2_registered": old_r2, "r2_delta": r2 - old_r2})
            summary = ae.summarise()
            summary_path = Path("comparison.csv")
            summary.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))
            summary_path.unlink(missing_ok=True)
        print("Logged run to MLflow")
    except Exception as exc:  # noqa: BLE001 - tracking must never fail the training run
        print(f"MLflow logging skipped: {type(exc).__name__}: {exc}")

    # ---- promotion -----------------------------------------------------------
    promoted = False
    if args.promote and improved:
        target = write_version(cfg, ae, best, new_version, df, dataset_path,
                               len(x_tr), len(x_te), r2, rmse)
        promoted = True
        print(f"Wrote new version -> {target}")
    elif args.promote:
        print("Not promoted: no improvement over the registered version.")
    else:
        print("Dry run (no --promote): registry unchanged.")

    report = markdown_report(cfg, old_version, old_r2, new_version, r2, rmse,
                             best.model_name, improved, promoted, len(x_tr), len(x_te))
    print("\n" + report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
