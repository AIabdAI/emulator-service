"""Generate the STAND-IN datasets this service is developed against.

The three sibling projects (battery-emulator, frame-emulator, pv-emulator) produce the
real datasets. When their artifacts are not present, this script synthesises small
tabular datasets from closed-form response surfaces so that the service is still built
and tested against *real* AutoEmulate emulators trained by the *real* pipeline.

The response surfaces are smooth analytic functions with plausible units and ranges.
They are **not** physics. Every model trained from them carries `"stand_in": true` in
its manifest, and the API surfaces that flag — a stand-in must never be mistaken for a
validated scientific emulator.

    python training/make_standin_datasets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

SEED = 20260827
DEFAULT_OUT = Path(__file__).resolve().parent / "data"


def _sample(bounds: dict[str, tuple[float, float]], n: int, seed: int) -> pd.DataFrame:
    """Latin hypercube over the parameter box — even coverage of the training domain."""
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    unit = sampler.random(n)
    lower = [low for low, _ in bounds.values()]
    upper = [high for _, high in bounds.values()]
    scaled = qmc.scale(unit, lower, upper)
    return pd.DataFrame(scaled, columns=list(bounds))


def battery_capacity_fade(n: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Stand-in for a PyBaMM capacity-fade study: % capacity lost after 500 cycles."""
    bounds = {
        "c_rate": (0.2, 3.0),
        "temperature": (5.0, 45.0),
        "depth_of_discharge": (0.2, 1.0),
    }
    df = _sample(bounds, n, seed)
    rng = np.random.default_rng(seed)

    # Arrhenius-flavoured temperature term + superlinear rate and DoD penalties.
    arrhenius = np.exp(-2400.0 * (1.0 / (df.temperature + 273.15) - 1.0 / 298.15))
    fade = (
        3.1 * arrhenius
        + 2.4 * df.c_rate**1.35
        + 4.6 * df.depth_of_discharge**2
        + 0.55 * df.c_rate * df.depth_of_discharge
    )
    df["capacity_fade"] = fade + rng.normal(0.0, 0.05, size=len(df))
    return df


def frame_peak_drift(n: int = 400, seed: int = SEED + 1) -> pd.DataFrame:
    """Stand-in for an OpenSees pushover study: peak interstorey drift ratio (%)."""
    bounds = {
        "pga": (0.05, 0.80),
        "yield_strength": (250.0, 500.0),
        "storey_mass": (40.0, 120.0),
        "damping_ratio": (0.01, 0.08),
    }
    df = _sample(bounds, n, seed)
    rng = np.random.default_rng(seed)

    period = 0.09 * np.sqrt(df.storey_mass) * (400.0 / df.yield_strength) ** 0.3
    drift = (
        1.85 * df.pga * period / (df.damping_ratio + 0.05) ** 0.45
        - 0.0016 * (df.yield_strength - 250.0)
        + 0.35
    )
    df["peak_drift_ratio"] = np.clip(drift, 0.05, None) + rng.normal(
        0.0, 0.01, size=len(df)
    )
    return df


DATASETS = {
    "battery_capacity_fade": battery_capacity_fade,
    "frame_peak_drift": frame_peak_drift,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=400, help="rows per dataset")
    parser.add_argument(
        "--only", choices=sorted(DATASETS), help="generate a single dataset"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else sorted(DATASETS)
    for name in names:
        frame = DATASETS[name](n=args.n)
        path = args.out_dir / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.8g")
        print(f"wrote {path} ({len(frame)} rows, {len(frame.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
