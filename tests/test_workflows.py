"""Validate the GitHub Actions workflow files.

These exist because a workflow YAML error does not fail loudly the way a code error
does: GitHub accepts the push, starts a run, and reports only "This run likely failed
because of a workflow file issue" with no line number. A malformed workflow is
effectively an untested file unless something like this parses it.

This caught a real bug. `run: docker images ... --format "Image size: {{.Size}}"` is a
*plain* YAML scalar containing ": ", so the parser read it as a nested mapping and the
whole CI workflow failed to load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.y*ml"))


def test_there_are_workflows_to_check():
    assert WORKFLOWS, f"no workflow files found under {WORKFLOW_DIR}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_parses_as_yaml(path: Path):
    yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_has_the_required_shape(path: Path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    assert doc.get("name"), f"{path.name} has no top-level 'name'"
    # PyYAML resolves the bare key `on` to the boolean True under YAML 1.1.
    assert "on" in doc or True in doc, f"{path.name} declares no trigger"
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{path.name} declares no jobs"

    for job_name, job in jobs.items():
        assert job.get("runs-on"), f"{path.name}:{job_name} has no runs-on"
        steps = job.get("steps")
        assert isinstance(steps, list) and steps, f"{path.name}:{job_name} has no steps"
        for i, step in enumerate(steps):
            assert isinstance(step, dict), (
                f"{path.name}:{job_name} step {i} parsed as {type(step).__name__}, "
                "not a mapping -- usually an unquoted ': ' inside a plain scalar"
            )
            assert "uses" in step or "run" in step, (
                f"{path.name}:{job_name} step {i} ({step.get('name')}) has neither "
                f"'uses' nor 'run'; keys were {sorted(step)}"
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_run_step_collapses_into_a_mapping(path: Path):
    """A `run:` that parses to anything but a string means the YAML was misread."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in doc["jobs"].items():
        for i, step in enumerate(job["steps"]):
            if "run" in step:
                assert isinstance(step["run"], str), (
                    f"{path.name}:{job_name} step {i} has a non-string 'run' "
                    f"({type(step['run']).__name__}) -- quote it or use a block scalar"
                )
