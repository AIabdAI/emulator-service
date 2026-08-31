# Model manifest schema

Every model in the registry is a **directory** containing exactly two things:

```
registry/<model_id>/<version>/
├── emulator          # the serialised AutoEmulate emulator (opaque to the service)
└── manifest.json     # this schema
```

The manifest is the contract between the projects that *train* emulators and the
service that *serves* them. The service refuses to load a model whose manifest does not
validate, and refuses to score inputs that fall outside the bounds the manifest
declares.

## Why the bounds live in the manifest

An emulator queried outside its training domain does not fail — it extrapolates, and
returns a confident-looking number that is simply wrong. That is the single most
dangerous failure mode of a surrogate model in production. Putting the training-domain
bounds in the manifest and enforcing them at the API boundary makes that failure
*impossible to reach* rather than merely documented.

## Schema

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | yes | Registry-unique id, `^[a-z0-9][a-z0-9_-]*$` |
| `version` | string | yes | Semantic version, `MAJOR.MINOR.PATCH` |
| `project` | string | yes | Project of origin (e.g. `battery-emulator`) |
| `description` | string | no | One line, human-facing |
| `autoemulate_version` | string | yes | Version the artifact was serialised with |
| `emulator_model` | string | yes | Winning model class (e.g. `GaussianProcessMatern32`) |
| `training_date` | string (ISO 8601) | yes | UTC date the emulator was fitted |
| `dataset_hash` | string | yes | SHA-256 of the training dataset file |
| `n_train` / `n_test` | integer | yes | Split sizes |
| `inputs` | array of Input | yes | Ordered; the order **is** the feature order |
| `output` | Output | yes | This registry serves one named output per model |
| `metrics` | Metrics | yes | Held-out performance |
| `artifact` | string | no | Filename of the serialised emulator (default `emulator`) |

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Feature name used in the request payload |
| `unit` | string | yes | Physical unit; `-` for dimensionless |
| `min` / `max` | number | yes | Training-domain bounds; `min < max` enforced |
| `description` | string | no | |

### Output

| Field | Type | Required |
|---|---|---|
| `name` | string | yes |
| `unit` | string | yes |
| `description` | string | no |

### Metrics

| Field | Type | Required | Description |
|---|---|---|---|
| `r2` | number | yes | Coefficient of determination on held-out data |
| `rmse` | number | yes | Root mean squared error, in the output's unit |

## Example

```json
{
  "model_id": "battery-capacity",
  "version": "1.0.0",
  "project": "battery-emulator",
  "description": "Discharge capacity of a Li-ion cell from design and operating conditions",
  "autoemulate_version": "1.2.1",
  "emulator_model": "GaussianProcessMatern32",
  "training_date": "2026-08-30",
  "dataset_hash": "9f2c...",
  "n_train": 480,
  "n_test": 120,
  "inputs": [
    {"name": "pos_electrode_thickness", "unit": "m", "min": 5.292e-05, "max": 9.828e-05},
    {"name": "neg_electrode_thickness", "unit": "m", "min": 5.964e-05, "max": 1.1076e-04},
    {"name": "pos_particle_radius", "unit": "m", "min": 3.654e-06, "max": 6.786e-06},
    {"name": "c_rate", "unit": "1/h", "min": 0.5, "max": 3.0},
    {"name": "ambient_temperature_C", "unit": "degC", "min": 5.0, "max": 45.0}
  ],
  "output": {"name": "capacity_Ah", "unit": "A.h"},
  "metrics": {"r2": 0.9749, "rmse": 0.2693},
  "artifact": "emulator"
}
```

## Versioning rules

* `model_id` identifies a *conceptual* model (`battery-capacity`); `version` identifies
  one trained artifact of it.
* A version directory is **immutable**. Retraining always writes a new version
  directory; the training pipeline refuses to overwrite an existing one.
* The API's `/models/{model_id}` serves the **highest** semantic version by default;
  any specific version can be addressed with `?version=1.0.0`.

## Validation performed at startup

1. `manifest.json` parses as JSON and satisfies the Pydantic model.
2. `version` and the parent directory name agree.
3. Every input has `min < max`.
4. The artifact file named by `artifact` exists.
5. The emulator deserialises, and a probe prediction at the *midpoint* of the declared
   bounds returns a finite mean and a non-negative variance.

A model that fails any check is skipped with a named, actionable log message; the rest
of the registry still loads. A registry with zero loadable models starts the service in
a state where `/health` reports `degraded`, so an orchestrator can act on it.
