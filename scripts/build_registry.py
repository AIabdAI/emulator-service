"""Package emulators from the sibling projects into this service's registry.

This is a **build-time** tool, not part of the serving path. It may read the sibling
projects' configs and datasets; the service itself never does.

Input bounds in the manifest are taken from the **actual min/max of each input column
in the training dataset**, not from the config's nominal ranges. That is the honest
definition of the training domain: it is where the emulator has actually seen data, and
it needs no simulator import to compute.

Usage:
    python scripts/build_registry.py --projects ../battery-emulator ../pv-emulator
    python scripts/build_registry.py --synthetic     # stand-ins, if siblings are absent
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "registry"

#: Sibling output name -> registry model id. Anything not listed is skipped.
MODEL_IDS = {
    "capacity_Ah": "battery-capacity",
    "energy_Wh": "battery-energy",
    "max_temp_rise_K": "battery-temperature-rise",
    "specific_yield_kWh_per_kWp": "pv-specific-yield",
    "capacity_factor_pct": "pv-capacity-factor",
    "clipping_loss_pct": "pv-clipping-loss",
    "peak_base_shear_kN": "frame-base-shear",
    "drift_at_peak_pct": "frame-drift-at-peak",
    "initial_stiffness_kN_per_m": "frame-initial-stiffness",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_from_project(project: Path, version: str, force: bool) -> list[str]:
    """Package every serialised emulator from one sibling project."""
    cfg_path = project / "config" / "config.yaml"
    best_path = project / "results" / "best_models.json"
    if not cfg_path.is_file():
        print(f"  ! {project.name}: no config/config.yaml, skipping")
        return []
    if not best_path.is_file():
        print(f"  ! {project.name}: no results/best_models.json -- run training first, skipping")
        return []

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))

    dataset_rel = cfg["paths"]["dataset"]
    dataset = Path(dataset_rel)
    if not dataset.is_absolute():
        dataset = project / dataset_rel
    if not dataset.is_file():
        print(f"  ! {project.name}: dataset {dataset} not found, skipping")
        return []

    df = pd.read_parquet(dataset)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    dataset_hash = sha256_file(dataset)

    params = [p["name"] for p in cfg["parameters"]]
    units = {p["name"]: str(p.get("unit", "-")) for p in cfg["parameters"]}
    out_units = {o["name"]: str(o.get("unit", "-")) for o in cfg["outputs"]}
    out_labels = {o["name"]: str(o.get("label", o["name"])) for o in cfg["outputs"]}

    created = []
    for output_name, info in best.items():
        model_id = MODEL_IDS.get(output_name)
        if model_id is None:
            print(f"  - {output_name}: no registry id mapped, skipping")
            continue

        # AutoEmulate.save() writes "<stem>.joblib" plus "<stem>_metadata.csv" and
        # reports the extensionless stem, so copy the whole file group.
        artifact_stem = project / info["artifact"]
        companions = sorted(artifact_stem.parent.glob(artifact_stem.name + "*"))
        if not companions:
            print(f"  ! {output_name}: no artifact files at {artifact_stem}*, skipping")
            continue

        target = REGISTRY / model_id / version
        if target.exists() and not force:
            print(f"  = {model_id} v{version} already exists (use --force to replace)")
            continue
        target.mkdir(parents=True, exist_ok=True)
        for src_file in companions:
            suffix = src_file.name[len(artifact_stem.name):]
            shutil.copy2(src_file, target / f"emulator{suffix}")

        manifest = {
            "model_id": model_id,
            "version": version,
            "project": project.name,
            "description": f"{out_labels.get(output_name, output_name)} "
                           f"emulated from {len(params)} design parameters",
            "autoemulate_version": pkg_version("autoemulate"),
            "emulator_model": info["best_model"],
            "training_date": dt.datetime.now(dt.UTC).date().isoformat(),
            "dataset_hash": dataset_hash,
            "n_train": int(info["n_train"]),
            "n_test": int(info["n_test"]),
            "inputs": [
                {
                    "name": p,
                    "unit": units.get(p, "-"),
                    "min": float(np.min(df[p])),
                    "max": float(np.max(df[p])),
                    "description": "training-domain bounds taken from the sampled dataset",
                }
                for p in params
            ],
            "output": {
                "name": output_name,
                "unit": out_units.get(output_name, "-"),
                "description": out_labels.get(output_name, output_name),
            },
            "metrics": {
                "r2": round(float(info["r2_test"]), 6),
                "rmse": round(float(info["rmse_test"]), 6),
            },
            "artifact": "emulator",
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  + {model_id} v{version}  ({info['best_model']}, R2={info['r2_test']:.4f})")
        created.append(model_id)
    return created


def build_synthetic(version: str, force: bool) -> list[str]:
    """Train two small stand-in emulators on synthetic data, using the real AutoEmulate API.

    Used only when the sibling projects are unavailable, so the service is still
    developed against genuinely serialised AutoEmulate objects. Manifests are clearly
    labelled as stand-ins.
    """
    from autoemulate import AutoEmulate
    from sklearn.model_selection import train_test_split

    print("Building SYNTHETIC stand-in models (clearly labelled in their manifests)")
    rng = np.random.default_rng(0)
    created = []

    specs = [
        ("synthetic-smooth", "y_smooth", "-",
         lambda x: np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.2 * x[:, 2]),
        ("synthetic-linear", "y_linear", "-",
         lambda x: 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2]),
    ]
    for model_id, output_name, unit, fn in specs:
        target = REGISTRY / model_id / version
        if target.exists() and not force:
            print(f"  = {model_id} v{version} already exists")
            continue

        x = rng.uniform(0.0, 1.0, size=(200, 3))
        y = fn(x).reshape(-1, 1)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
        ae = AutoEmulate(
            x_tr, y_tr, test_data=(x_te, y_te),
            models=["GaussianProcessRBF"], n_iter=3, n_splits=2,
            random_seed=0, log_level="error",
        )
        best = ae.best_result()
        target.mkdir(parents=True, exist_ok=True)
        ae.save(best, target / "emulator", use_timestamp=False)

        def metric(metrics, name):
            for k, v in metrics.items():
                if getattr(k, "name", str(k)) == name:
                    return float(v[0] if isinstance(v, tuple) else v)
            return float("nan")

        manifest = {
            "model_id": model_id,
            "version": version,
            "project": "synthetic-standin",
            "description": "STAND-IN model trained on synthetic data, not a physical emulator",
            "autoemulate_version": pkg_version("autoemulate"),
            "emulator_model": best.model_name,
            "training_date": dt.datetime.now(dt.UTC).date().isoformat(),
            "dataset_hash": "synthetic-" + hashlib.sha256(x.tobytes()).hexdigest()[:32],
            "n_train": int(len(x_tr)),
            "n_test": int(len(x_te)),
            "inputs": [
                {"name": f"x{i + 1}", "unit": "-", "min": 0.0, "max": 1.0} for i in range(3)
            ],
            "output": {"name": output_name, "unit": unit},
            "metrics": {
                "r2": round(metric(best.test_metrics, "r2"), 6),
                "rmse": round(metric(best.test_metrics, "rmse"), 6),
            },
            "artifact": "emulator",
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  + {model_id} v{version} (stand-in, {best.model_name})")
        created.append(model_id)
    return created


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the model registry.")
    ap.add_argument("--projects", nargs="*", default=[], help="Paths to sibling project repos")
    ap.add_argument("--version", default="1.0.0", help="Semantic version to write")
    ap.add_argument("--synthetic", action="store_true", help="Also build stand-in models")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing version dir")
    args = ap.parse_args(argv)

    REGISTRY.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for p in args.projects:
        project = Path(p).resolve()
        print(f"Project {project}")
        created += build_from_project(project, args.version, args.force)

    if args.synthetic or not created:
        if not created and not args.synthetic:
            print("No sibling artifacts found -- falling back to synthetic stand-ins.")
        created += build_synthetic(args.version, args.force)

    n_versions = len(list(REGISTRY.glob("*/*/manifest.json")))
    print(f"\nRegistry now contains {n_versions} model version(s)")
    return 0 if created else 1


if __name__ == "__main__":
    sys.exit(main())
