# Model manifest schema (v1)

Every model in the registry is a **directory**, not a file. The directory carries the
serialized AutoEmulate emulator plus a `manifest.json` that describes it well enough for
the service to validate requests, report provenance and refuse to serve nonsense —
without ever importing the simulator that produced the training data.

## Registry layout

```
registry/
└── <model_id>/                  # stable, kebab-case identity of a modelled quantity
    └── <version>/               # semantic version, e.g. 1.0.0 — immutable once written
        ├── manifest.json        # this schema
        ├── model.joblib         # joblib-serialized autoemulate Emulator
        └── model_metadata.csv   # optional: AutoEmulate Result metadata (params/metrics)
```

Rules:

- A `<model_id>/<version>/` directory is **immutable**. Retraining writes a *new* version
  directory; it never overwrites an existing one (`training/retrain.py` enforces this).
- `manifest.json` is validated at service startup. A malformed manifest fails loudly with
  a message naming the file and the offending field — it does not silently skip the model.
- The registry is the single source of truth for input bounds. The API derives its
  validation from the manifest, so a model cannot be queried outside its training domain.

## Serialization contract (AutoEmulate 1.2.1)

Verified against the installed package (`autoemulate.core.save.ModelSerialiser`,
`autoemulate.core.compare.AutoEmulate`):

| Operation | Call | Notes |
|---|---|---|
| Save | `AutoEmulate.save(result_or_emulator, path, use_timestamp=False)` | `joblib.dump` of the `Emulator`; writes `<name>_metadata.csv` alongside when given a `Result` |
| Load | `AutoEmulate.load_model(path)` | **staticmethod** — no `AutoEmulate` instance, no simulator, no training data needed |
| Predict + UQ | `emulator.predict_mean_and_variance(x)` | returns `(mean, variance)`; `variance is None` when the emulator does not support UQ |

`predict_mean_and_variance` is defined on `autoemulate.emulators.base.Emulator` and
specialised by `DeterministicEmulator` (variance `None`) and `ProbabilisticEmulator`
(variance tensor). Serving against this one method — rather than `predict()`, whose return
type varies between a tensor and a `torch.distributions.Distribution` — is what lets the
API return uncertainty uniformly for every emulator family.

Inputs and outputs are `torch.Tensor` of shape `(n_batch, n_features)` /
`(n_batch, n_targets)`, float64 by default.

## `manifest.json` fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | `int` | yes | Manifest schema version. Currently `1`. |
| `model_id` | `str` | yes | Stable id, `^[a-z0-9][a-z0-9-]*$`. Must equal the parent directory name. |
| `version` | `str` | yes | Semantic version `MAJOR.MINOR.PATCH`. Must equal the directory name. |
| `project` | `str` | yes | Sibling project of origin, e.g. `battery-emulator`. |
| `description` | `str` | no | One line, human-facing. |
| `autoemulate_version` | `str` | yes | Version the artifact was serialized with. Startup warns on mismatch with the installed runtime. |
| `training_date` | `str` | yes | ISO-8601 date or datetime, UTC. |
| `artifact` | `object` | yes | See [Artifact](#artifact). |
| `inputs` | `array` | yes | Ordered; **column order is the tensor column order**. See [Input parameter](#input-parameter). |
| `output` | `object` | yes | See [Output](#output). |
| `metrics` | `object` | yes | Held-out metrics. `r2` and `rmse` required. |
| `dataset` | `object` | yes | See [Dataset](#dataset). |
| `stand_in` | `bool` | no | `true` marks a synthetic development artifact, not a real scientific emulator. Surfaced by the API. Defaults to `false`. |

### Artifact

| Field | Type | Required | Description |
|---|---|---|---|
| `filename` | `str` | yes | Relative to the version directory. Must exist. Must not escape it. |
| `format` | `str` | yes | `joblib` (only supported value). |
| `emulator_class` | `str` | yes | e.g. `GaussianProcessExact`. Informational; checked against the loaded object at startup. |
| `supports_uq` | `bool` | yes | Whether `predict_mean_and_variance` returns a variance. Cross-checked against the loaded emulator. |
| `sha256` | `str` | no | Hex digest of the artifact file, verified at load when present. |

### Input parameter

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Request field name. Unique within the manifest. |
| `unit` | `str` | yes | Physical unit, or `"dimensionless"`. Echoed in errors and `/models/{id}`. |
| `min` | `number` | yes | Inclusive lower bound of the training domain. |
| `max` | `number` | yes | Inclusive upper bound. Must be `> min`. |
| `description` | `str` | no | One line. |

Bounds are the **training domain**, not physical limits. An emulator queried outside the
region it was fitted on is not merely imprecise — it is confidently, silently wrong. The
API therefore rejects out-of-domain rows with `422` rather than extrapolating.

### Output

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Predicted quantity. |
| `unit` | `str` | yes | Physical unit, or `"dimensionless"`. |
| `description` | `str` | no | One line. |

Single-output emulators only in schema v1. A multi-output model would extend this to an
`outputs` array; the loader rejects an artifact whose predicted width is not 1 so the
mismatch cannot pass silently.

### Metrics

| Field | Type | Required | Description |
|---|---|---|---|
| `r2` | `number` | yes | Held-out coefficient of determination. |
| `rmse` | `number` | yes | Held-out RMSE, in the output unit. `>= 0`. |
| `n_test` | `int` | no | Held-out set size. |

Extra numeric keys are permitted and passed through.

### Dataset

| Field | Type | Required | Description |
|---|---|---|---|
| `hash` | `str` | yes | `sha256:<hex>` of the training file. Ties the model to exact data. |
| `n_train` | `int` | no | Training row count. |
| `path` | `str` | no | DVC-tracked path the hash refers to. |

## Example

```json
{
  "schema_version": 1,
  "model_id": "battery-capacity-fade",
  "version": "1.0.0",
  "project": "battery-emulator",
  "description": "Stand-in emulator for capacity fade after 500 cycles.",
  "autoemulate_version": "1.2.1",
  "training_date": "2026-08-27",
  "stand_in": true,
  "artifact": {
    "filename": "model.joblib",
    "format": "joblib",
    "emulator_class": "GaussianProcessExact",
    "supports_uq": true,
    "sha256": "6f1e..."
  },
  "inputs": [
    {"name": "c_rate", "unit": "1/h", "min": 0.2, "max": 3.0,
     "description": "Charge/discharge rate."},
    {"name": "temperature", "unit": "degC", "min": 5.0, "max": 45.0,
     "description": "Ambient cell temperature."}
  ],
  "output": {"name": "capacity_fade", "unit": "percent",
             "description": "Capacity lost after 500 cycles."},
  "metrics": {"r2": 0.987, "rmse": 0.42, "n_test": 80},
  "dataset": {"hash": "sha256:9ab3...", "n_train": 320,
              "path": "training/data/battery_capacity_fade.csv"}
}
```

## Validation and failure behaviour

At startup the loader, for every `registry/*/*/`:

1. reads and JSON-parses `manifest.json`;
2. validates it against this schema (Pydantic v2, `extra="forbid"` on nested objects);
3. checks `model_id` / `version` match the directory names;
4. checks the artifact file exists, and its `sha256` if declared;
5. `joblib`-loads the emulator via `AutoEmulate.load_model`;
6. runs one probe prediction at the bounds midpoint to confirm the emulator accepts
   `len(inputs)` features, returns width 1, and agrees with `supports_uq`.

Any failure raises `RegistryError` naming **the path and the field**, e.g.

```
registry/battery-capacity-fade/1.0.0/manifest.json: inputs[1]: max (5.0) must be greater
than min (45.0)
```

A model that fails validation is never served. Whether one bad model takes the whole
service down or is quarantined is a deployment choice: `strict=True` (the default, and
what `docker compose` uses) fails startup, so a broken registry cannot masquerade as a
healthy one; `strict=False` loads what it can and reports the failures on `/health`.
