"""Check the shared Nautilus launcher's stopping options and saved output.

Nautilus can save evaluated points to its checkpoint even when it cannot
assign posterior weights. Run the launcher with two analytic likelihoods
to verify that the checkpoint and the posterior chain report different
quantities. Run this file with pytest from the activated Cocoa environment.
"""

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml


def test_nautilus_effective_sample_target():
    """Read the shared target and check explicit options for a bounded test."""
    core = Path(__file__).resolve().parents[1]
    code = (
        "from pathlib import Path\n"
        "from cocoa_hybrid_sampling import parser_for\n"
        "parser = parser_for(mode='nautilus', project=Path('.'), example=1)\n"
        "defaults = parser.parse_args(args=[])\n"
        "assert defaults.neff == 15000\n"
        "assert defaults.maxfeval == 5000000\n"
        "options = parser.parse_args(args=['--neff', '50', '--maxfeval', '100'])\n"
        "assert options.neff == 50 and options.maxfeval == 100\n"
    )
    command = [
        sys.executable,
        "-c",
        code,
    ]
    completed = subprocess.run(
        args=command,
        cwd=core, capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("sample_case", ["discarded", "rejected", "retained"])
def test_nautilus_posterior_output(tmp_path, sample_case):
    """Check the checkpoint and chain from a one-batch Nautilus run.

    A flat likelihood with a high stopping fraction discards the whole
    exploration batch and leaves zero effective samples. A likelihood
    finite only at the supplied start rejects every prior draw and leaves
    undefined weights. A flat likelihood that keeps exploration verifies
    the usual chain output for finite positive effective sample size.

    Arguments:
        tmp_path = directory for the YAML, sampler checkpoint and chain.
        sample_case = whether the budget discards, rejects or keeps the batch.
    Returns:
        None when the checkpoint, run summary and chain agree.
    """
    likelihood = "lambda x, y: 0.0"
    live_fraction = "0.01"
    if sample_case == "rejected":
        likelihood = "lambda x, y: 0.0 if x == 0.0 and y == 0.0 else -float('inf')"
    if sample_case == "discarded":
        # Ten live points hold one tenth of a flat 100-draw batch's evidence.
        # A threshold of one half then discards that batch as exploration.
        live_fraction = "0.5"
    config = {
        "likelihood": {"analytic": {"external": likelihood}},
        "params": {
            "x": {"prior": {"min": -1.0, "max": 1.0}},
            "y": {"prior": {"min": -1.0, "max": 1.0}},
        },
        "sampler": {"evaluate": {"override": {"x": 0.0, "y": 0.0}}},
    }
    source = tmp_path / "input.yaml"
    source.write_text(yaml.safe_dump(config, sort_keys=False))
    core = Path(__file__).resolve().parents[1]
    # The regular launcher runs in its own interpreter and owns the Cobaya
    # model. The test supplies a cheap YAML instead of survey data.
    command = [
        sys.executable,
        "-c",
        "from cocoa_hybrid_sampling import run\n"
        "run(mode='nautilus', project='.', hybrid=False)",
        "--input", str(source),
        "--root", str(tmp_path),
        "--outroot", "analytic",
        "--nlive", "10",
        "--maxfeval", "100",
        "--flive", live_fraction,
    ]
    run = subprocess.run(command, cwd=core, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr

    chains = tmp_path / "chains"
    record = yaml.safe_load((chains / "analytic.yaml").read_text())
    assert record["evaluations"] == 100
    assert not record["converged"]
    with h5py.File(chains / "analytic_checkpoint.hdf5", "r") as checkpoint:
        sampler = checkpoint["sampler"]
        scores = sampler["log_l_0"][:]
        discarded_exploration = sampler.attrs["_discard_exploration"]
    assert scores.shape == (100,)
    assert discarded_exploration == (sample_case == "discarded")
    chain = chains / "analytic.1.txt"
    parameter_names = chains / "analytic.paramnames"
    if sample_case == "retained":
        assert record["posterior_samples"] == 100
        assert np.isclose(record["effective_samples"], 100.0)
        assert np.isfinite(np.loadtxt(chain)).all()
        assert parameter_names.is_file()
    else:
        if sample_case == "discarded":
            assert np.isfinite(scores).all()
        else:
            assert np.isneginf(scores).all()
        assert record["posterior_samples"] == 0
        assert record["effective_samples"] == 0.0
        assert record["log_evidence"] is None
        assert not chain.exists()
        assert not parameter_names.exists()


def test_nautilus_resumes_checkpoint(tmp_path):
    """Resume a bounded sample with the default likelihood-call budget.

    The first batch has positive weights. Its chain must be replaced by
    the resumed weights, and a changed exploration setting must be refused
    before it touches the checkpoint.
    """
    config = {
        "likelihood": {"analytic": {"external": "lambda x, y: 0.0"}},
        "params": {
            "x": {"prior": {"min": -1.0, "max": 1.0}},
            "y": {"prior": {"min": -1.0, "max": 1.0}},
        },
        "sampler": {"evaluate": {"override": {"x": 0.0, "y": 0.0}}},
    }
    source = tmp_path / "input.yaml"
    source.write_text(yaml.safe_dump(config, sort_keys=False))
    core = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-c",
        "from cocoa_hybrid_sampling import run\n"
        "run(mode='nautilus', project='.', hybrid=False)",
        "--input", str(source),
        "--root", str(tmp_path),
        "--outroot", "analytic",
        "--nlive", "10",
        "--flive", "0.09",
        "--neff", "150",
    ]
    bounded = subprocess.run(command + ["--maxfeval", "100"], cwd=core,
                             capture_output=True, text=True, timeout=30)
    assert bounded.returncode == 0, bounded.stdout + bounded.stderr
    prefix = tmp_path / "chains" / "analytic"
    checkpoint = prefix.with_name("analytic_checkpoint.hdf5")
    chain = prefix.with_suffix(".1.txt")
    record = prefix.with_suffix(".yaml")
    first = yaml.safe_load(record.read_text())
    assert first["evaluations"] == 100 and not first["converged"]
    assert len(np.loadtxt(chain)) == first["posterior_samples"] == 100

    original_checkpoint = checkpoint.read_bytes()
    invalid = subprocess.run(command + ["--resume", "--nlive", "11"], cwd=core,
                             capture_output=True, text=True, timeout=30)
    assert invalid.returncode != 0
    assert "nlive was 10, requested 11" in invalid.stdout + invalid.stderr
    assert checkpoint.read_bytes() == original_checkpoint

    resumed = subprocess.run(command + ["--resume"], cwd=core, capture_output=True,
                             text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Resuming nautilus run" in resumed.stdout
    result = yaml.safe_load(record.read_text())
    assert result["converged"] and result["effective_samples"] >= 150
    assert result["maxfeval"] == 5_000_000 and result["evaluations"] > 100
    samples = np.loadtxt(chain)
    assert len(samples) == result["posterior_samples"]
    np.testing.assert_allclose(samples[:, 0].sum(), 1.0)
    with h5py.File(checkpoint, "r") as saved:
        assert saved["sampler"].attrs["n_like"] == result["evaluations"]
