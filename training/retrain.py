"""Retrain an emulator from a conforming dataset and propose it to the registry.

Given a dataset config (`training/configs/*.yaml`) and a CSV/Parquet table, this:

1. hashes the dataset so the resulting model is tied to exact data;
2. runs the AutoEmulate model comparison on a held-out split;
3. logs params, metrics and artifacts to MLflow;
4. compares the champion against the incumbent registry version;
5. **on improvement only**, writes a new versioned model directory — never overwriting
   an existing version;
6. emits a markdown report (for CML to post on a PR) and a JSON summary.

The promote/reject decision is a *recommendation*. In CI this runs against a scratch
registry and nothing is committed: a human merges.

    python training/retrain.py --config training/configs/battery_capacity_fade.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from api.manifest import (  # noqa: E402
    ArtifactSpec,
    DatasetSpec,
    InputParameter,
    Manifest,
    MetricsSpec,
    OutputSpec,
    RegistryError,
    load_manifest,
    sha256_file,
    write_manifest,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retrain")

ARTIFACT_NAME = "model.joblib"
METADATA_NAME = "model_metadata.csv"


# ------------------------------------------------------------------- config


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputConfig(_Strict):
    name: str
    unit: str
    description: str | None = None
    min: float | None = None
    max: float | None = None


class OutputConfig(_Strict):
    name: str
    unit: str
    description: str | None = None


class TrainingConfig(_Strict):
    models: list[str] | None = None
    n_splits: int = Field(default=4, ge=2)
    n_iter: int = Field(default=6, ge=1)
    n_bootstraps: int | None = Field(default=None, ge=1)
    only_probabilistic: bool = False
    random_seed: int = 42


class PromotionConfig(_Strict):
    metric: str = "r2"
    min_improvement: float = Field(
        default=0.0,
        description="New metric must beat the incumbent by at least this margin.",
    )
    version_bump: str = Field(default="minor", pattern="^(major|minor|patch)$")


class DatasetConfig(_Strict):
    path: str
    format: str | None = None


class RetrainConfig(_Strict):
    model_id: str
    project: str
    description: str | None = None
    stand_in: bool = False
    dataset: DatasetConfig
    inputs: list[InputConfig] = Field(min_length=1)
    output: OutputConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    mlflow_experiment: str | None = None


def load_config(path: Path) -> RetrainConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path}: not valid YAML: {exc}") from exc
    try:
        return RetrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"{path}: invalid dataset config:\n{exc}") from exc


# ------------------------------------------------------------------- dataset


def load_dataset(path: Path, fmt: str | None) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(
            f"dataset not found: {path}\n" f"If it is DVC-tracked, run `dvc pull` first."
        )
    fmt = fmt or ("parquet" if path.suffix in {".parquet", ".pq"} else "csv")
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise SystemExit(f"unsupported dataset format {fmt!r} (expected csv or parquet)")


def resolve_columns(frame: pd.DataFrame, config: RetrainConfig) -> None:
    expected = [item.name for item in config.inputs] + [config.output.name]
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise SystemExit(
            f"dataset {config.dataset.path} is missing required columns {missing}. "
            f"It has: {list(frame.columns)}"
        )
    subset = frame[expected]
    if subset.isna().any().any():
        bad = subset.columns[subset.isna().any()].tolist()
        raise SystemExit(f"dataset has missing values in columns {bad}")


def build_bounds(frame: pd.DataFrame, config: RetrainConfig) -> list[InputParameter]:
    """Training-domain bounds: the observed data range, unless pinned in the config.

    Deriving bounds from the data is the honest default — the emulator's trustworthy
    region is exactly where it saw examples, not where physics happens to be defined.
    """
    parameters = []
    for item in config.inputs:
        column = frame[item.name]
        low = item.min if item.min is not None else float(column.min())
        high = item.max if item.max is not None else float(column.max())
        parameters.append(
            InputParameter(
                name=item.name,
                unit=item.unit,
                min=round(float(low), 10),
                max=round(float(high), 10),
                description=item.description,
            )
        )
    return parameters


# ------------------------------------------------------------------ registry


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def incumbent(registry_root: Path, model_id: str) -> Manifest | None:
    """Highest registered version of this model, or None if it is brand new."""
    model_dir = registry_root / model_id
    if not model_dir.is_dir():
        return None
    versions = sorted(
        (
            p.name
            for p in model_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").is_file()
        ),
        key=_version_key,
    )
    if not versions:
        return None
    return load_manifest(model_dir / versions[-1] / "manifest.json")


def next_version(current: str | None, bump: str) -> str:
    if current is None:
        return "1.0.0"
    major, minor, patch = _version_key(current)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


# ------------------------------------------------------------------ training


def run_comparison(frame: pd.DataFrame, config: RetrainConfig):
    """Run the AutoEmulate comparison and return `(autoemulate, best_result)`."""
    from autoemulate import AutoEmulate

    x = frame[[item.name for item in config.inputs]].to_numpy(dtype="float64")
    y = frame[[config.output.name]].to_numpy(dtype="float64")

    settings = config.training
    logger.info(
        "running AutoEmulate on %d rows x %d inputs (models=%s)",
        x.shape[0],
        x.shape[1],
        settings.models or "all",
    )
    autoemulate = AutoEmulate(
        x,
        y,
        models=settings.models,
        n_splits=settings.n_splits,
        n_iter=settings.n_iter,
        n_bootstraps=settings.n_bootstraps,
        only_probabilistic=settings.only_probabilistic,
        random_seed=settings.random_seed,
        log_level="error",
    )
    return autoemulate, autoemulate.best_result(config.promotion.metric)


def test_metrics(result) -> dict[str, float]:
    """Flatten `Result.test_metrics` (keyed by Metric objects) to plain floats."""
    return {str(metric): float(value[0]) for metric, value in result.test_metrics.items()}


def probe_supports_uq(emulator, parameters: list[InputParameter]) -> bool:
    """Ask the fitted emulator itself whether it returns a variance.

    Tries float64 then float32: AutoEmulate fits some emulators (notably the exact GPs)
    in single precision, and a double input raises rather than being upcast. The serving
    loader performs the same negotiation, so what is recorded here is what will be served.
    """
    import torch

    midpoint = [[p.midpoint for p in parameters]]
    last_error: Exception | None = None
    for dtype in (torch.float64, torch.float32):
        try:
            with torch.inference_mode():
                _, variance = emulator.predict_mean_and_variance(
                    torch.tensor(midpoint, dtype=dtype)
                )
        except Exception as exc:
            last_error = exc
            continue
        return variance is not None
    raise SystemExit(
        f"the trained emulator could not be probed at the domain midpoint: {last_error}"
    )


# -------------------------------------------------------------------- MLflow


def default_tracking_uri() -> str:
    """Local fallback when no tracking server is configured.

    MLflow 3 has retired the plain-filesystem store, so the offline default is a local
    SQLite database — the same backend the `docker compose` tracking server uses, which
    keeps `MLFLOW_TRACKING_URI=http://localhost:5000` a pure config swap.
    """
    database = REPO_ROOT / "mlruns" / "mlflow.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + database.as_posix()


class Tracker:
    """Thin MLflow wrapper that degrades to a no-op if tracking is unavailable.

    A tracking server being down should not lose a training run; the model artifact and
    the report are the deliverables, the run record is a nice-to-have.
    """

    def __init__(self, experiment: str, uri: str | None, enabled: bool = True):
        self.enabled = enabled
        self.run_id: str | None = None
        self.run_uri: str | None = None
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(uri or default_tracking_uri())
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except Exception as exc:
            logger.warning("MLflow tracking disabled: %s", exc)
            self.enabled = False

    def __enter__(self) -> Tracker:
        if self._mlflow is not None:
            try:
                run = self._mlflow.start_run()
                self.run_id = run.info.run_id
                self.run_uri = self._mlflow.get_tracking_uri()
            except Exception as exc:
                logger.warning("MLflow run could not start: %s", exc)
                self._mlflow = None
                self.enabled = False
        return self

    def __exit__(self, *exc_info) -> None:
        if self._mlflow is not None and self.run_id is not None:
            try:
                self._mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow run could not be closed: %s", exc)

    def _safe(self, name: str, *args, **kwargs) -> None:
        if self._mlflow is None:
            return
        try:
            getattr(self._mlflow, name)(*args, **kwargs)
        except Exception as exc:
            logger.warning("MLflow %s failed: %s", name, exc)

    def log_params(self, params: dict[str, Any]) -> None:
        self._safe("log_params", {k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._safe("log_metrics", {k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._safe("set_tags", {k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._safe("log_artifact", str(path), artifact_path)

    def log_dir(self, path: Path, artifact_path: str | None = None) -> None:
        self._safe("log_artifacts", str(path), artifact_path)


# -------------------------------------------------------------------- report


def render_report(summary: dict[str, Any]) -> str:
    """Markdown for a CML PR comment: old vs new, and a recommendation."""
    decision = summary["decision"]
    icon = {"promote": "✅", "reject": "🛑", "new": "🆕"}[decision]
    old = summary["incumbent"]
    new = summary["candidate"]
    metric = summary["promotion_metric"]

    lines = [
        f"## {icon} Retrain report — `{summary['model_id']}`",
        "",
        f"**Recommendation: {summary['recommendation']}**",
        "",
        f"> {summary['rationale']}",
        "",
        "### Metrics (held-out)",
        "",
        "| Metric | Incumbent | Candidate | Δ |",
        "|---|---:|---:|---:|",
    ]

    metric_names = sorted(set(new["metrics"]) | set((old or {}).get("metrics", {})))
    for name in metric_names:
        new_value = new["metrics"].get(name)
        old_value = (old or {}).get("metrics", {}).get(name)
        delta = (
            f"{new_value - old_value:+.4f}"
            if new_value is not None and old_value is not None
            else "—"
        )
        marker = " ⭐" if name == metric else ""
        lines.append(
            f"| `{name}`{marker} "
            f"| {'—' if old_value is None else f'{old_value:.4f}'} "
            f"| {'—' if new_value is None else f'{new_value:.4f}'} "
            f"| {delta} |"
        )

    previous = old or {}

    def was(key: str) -> str:
        value = previous.get(key)
        return "—" if value is None else str(value)

    lines += [
        "",
        "### Provenance",
        "",
        "| | Incumbent | Candidate |",
        "|---|---|---|",
        f"| Version | {was('version')} | {new['version']} |",
        f"| Emulator | {was('emulator_class')} | {new['emulator_class']} |",
        f"| Trained | {was('training_date')} | {new['training_date']} |",
        f"| Rows | {was('n_train')} | {new['n_train']} |",
        f"| Dataset hash | `{was('dataset_hash')[:19]}` | `{new['dataset_hash'][:19]}` |",
        "",
    ]

    if summary.get("mlflow_run_id"):
        lines += [f"MLflow run: `{summary['mlflow_run_id']}`", ""]

    if decision == "promote":
        lines += [
            f"The candidate is written to `registry/{summary['model_id']}/"
            f"{new['version']}/`. **Nothing is auto-promoted**: review the diff and "
            "merge this PR to put it in front of traffic.",
        ]
    elif decision == "new":
        lines += [
            f"No incumbent — this is the first version of `{summary['model_id']}`. "
            "Review the metrics before merging.",
        ]
    else:
        lines += [
            f"The candidate did not beat the incumbent `{metric}` by the required "
            f"margin of {summary['min_improvement']}. No registry version was written.",
        ]

    if summary.get("stand_in"):
        lines += [
            "",
            "> ⚠️ This model is trained on a **stand-in** synthetic dataset, not "
            "validated simulator output.",
        ]

    lines += [
        "",
        "<sub>Generated by `training/retrain.py`. Automation proposes; a human "
        "merges.</sub>",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="dataset config YAML")
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "registry",
        help="registry root to compare against and write into",
    )
    parser.add_argument("--data", type=Path, help="override the config's dataset path")
    parser.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    parser.add_argument(
        "--report", type=Path, help="write the markdown report here (for CML)"
    )
    parser.add_argument("--json", type=Path, help="write the JSON summary here")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate and report, but never write to the registry",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_path = args.data or (REPO_ROOT / config.dataset.path)
    dataset_path = Path(dataset_path)

    frame = load_dataset(dataset_path, config.dataset.format)
    resolve_columns(frame, config)
    dataset_hash = sha256_file(dataset_path)
    parameters = build_bounds(frame, config)

    registry_root: Path = args.registry
    previous = incumbent(registry_root, config.model_id)
    candidate_version = next_version(
        previous.version if previous else None, config.promotion.version_bump
    )

    experiment = config.mlflow_experiment or f"emulator-{config.model_id}"
    with Tracker(experiment, args.mlflow_uri, enabled=not args.no_mlflow) as tracker:
        tracker.set_tags(
            {
                "model_id": config.model_id,
                "project": config.project,
                "stand_in": config.stand_in,
                "incumbent_version": previous.version if previous else "none",
                "candidate_version": candidate_version,
            }
        )
        tracker.log_params(
            {
                "dataset_path": str(dataset_path.relative_to(REPO_ROOT))
                if dataset_path.is_relative_to(REPO_ROOT)
                else str(dataset_path),
                "dataset_hash": dataset_hash,
                "n_rows": len(frame),
                "inputs": ",".join(p.name for p in parameters),
                "output": config.output.name,
                "autoemulate_version": pkg_version("autoemulate"),
                **{f"train_{k}": v for k, v in config.training.model_dump().items()},
                "promotion_metric": config.promotion.metric,
                "min_improvement": config.promotion.min_improvement,
            }
        )

        autoemulate, result = run_comparison(frame, config)
        metrics = test_metrics(result)
        logger.info("champion: %s  metrics=%s", result.model_name, metrics)

        tracker.set_tags({"champion_model": result.model_name})
        tracker.log_metrics(metrics)

        metric_name = config.promotion.metric
        if metric_name not in metrics:
            raise SystemExit(
                f"promotion metric {metric_name!r} not among evaluated metrics "
                f"{sorted(metrics)}"
            )
        candidate_score = metrics[metric_name]
        incumbent_score = (
            getattr(previous.metrics, metric_name, None) if previous else None
        )
        if incumbent_score is None and previous is not None:
            incumbent_score = getattr(previous.metrics, "__pydantic_extra__", {}).get(
                metric_name
            )

        higher_is_better = metric_name not in {"rmse", "mse", "mae", "msll", "crps"}
        if previous is None:
            decision, rationale = (
                "new",
                (f"First version of `{config.model_id}` — no incumbent to beat."),
            )
            promote = True
        else:
            margin = config.promotion.min_improvement
            if higher_is_better:
                promote = candidate_score >= incumbent_score + margin
                direction = "higher"
            else:
                promote = candidate_score <= incumbent_score - margin
                direction = "lower"
            decision = "promote" if promote else "reject"
            rationale = (
                f"Candidate `{metric_name}` = {candidate_score:.4f} vs incumbent "
                f"{incumbent_score:.4f} ({direction} is better, required margin "
                f"{margin}). "
                + ("Improvement met." if promote else "Improvement not met.")
            )

        supports_uq = probe_supports_uq(result.model, parameters)
        version_dir = registry_root / config.model_id / candidate_version
        written = False

        if promote and not args.dry_run:
            if version_dir.exists():
                raise RegistryError(
                    version_dir,
                    "version directory already exists; registry versions are immutable",
                )
            version_dir.mkdir(parents=True)
            saved = autoemulate.save(result, path=version_dir, use_timestamp=False)
            artifact_path = version_dir / ARTIFACT_NAME
            Path(f"{saved}.joblib").replace(artifact_path)
            metadata_source = Path(f"{saved}_metadata.csv")
            if metadata_source.is_file():
                metadata_source.replace(version_dir / METADATA_NAME)

            description = config.description
            if description:
                description = f"{description} [AutoEmulate: {result.model_name}]"

            manifest = Manifest(
                model_id=config.model_id,
                version=candidate_version,
                project=config.project,
                description=description,
                autoemulate_version=pkg_version("autoemulate"),
                training_date=dt.datetime.now(dt.UTC).date().isoformat(),
                stand_in=config.stand_in,
                artifact=ArtifactSpec(
                    filename=ARTIFACT_NAME,
                    format="joblib",
                    emulator_class=type(result.model).__name__,
                    supports_uq=supports_uq,
                    sha256=sha256_file(artifact_path),
                ),
                inputs=parameters,
                output=OutputSpec(**config.output.model_dump()),
                metrics=MetricsSpec(
                    r2=metrics["r2"],
                    rmse=metrics["rmse"],
                    **{k: v for k, v in metrics.items() if k not in {"r2", "rmse"}},
                ),
                dataset=DatasetSpec(
                    hash=f"sha256:{dataset_hash}",
                    n_train=len(frame),
                    path=str(dataset_path.relative_to(REPO_ROOT)).replace("\\", "/")
                    if dataset_path.is_relative_to(REPO_ROOT)
                    else str(dataset_path),
                ),
            )
            write_manifest(version_dir, manifest)
            written = True
            logger.info("wrote %s", version_dir)
            tracker.log_dir(version_dir, artifact_path="registry_candidate")
        elif promote:
            logger.info("--dry-run: would have written %s", version_dir)

        summary: dict[str, Any] = {
            "model_id": config.model_id,
            "project": config.project,
            "stand_in": config.stand_in,
            "decision": decision,
            "recommendation": {
                "promote": "PROMOTE",
                "new": "PROMOTE (first version)",
                "reject": "REJECT",
            }[decision],
            "rationale": rationale,
            "promotion_metric": metric_name,
            "min_improvement": config.promotion.min_improvement,
            "written": written,
            "registry_dir": str(version_dir) if written else None,
            "mlflow_run_id": tracker.run_id,
            "candidate": {
                "version": candidate_version,
                "metrics": metrics,
                "emulator_class": result.model_name,
                "supports_uq": supports_uq,
                "training_date": dt.datetime.now(dt.UTC).date().isoformat(),
                "n_train": len(frame),
                "dataset_hash": f"sha256:{dataset_hash}",
            },
            "incumbent": None
            if previous is None
            else {
                "version": previous.version,
                "metrics": previous.metrics.model_dump(exclude_none=True),
                "emulator_class": previous.artifact.emulator_class,
                "training_date": previous.training_date,
                "n_train": previous.dataset.n_train,
                "dataset_hash": previous.dataset.hash,
            },
        }

        report = render_report(summary)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
            tracker.log_artifact(args.report, "report")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            tracker.log_artifact(args.json, "report")

    # The report contains emoji; a Windows console defaults to cp1252 and would
    # otherwise fail on it after the work is already done.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
