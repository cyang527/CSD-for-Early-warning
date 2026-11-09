"""Capacity preparation and identification of the battery model coefficients.

MIT uses a full-record spline and full-record outlier statistics. Tongji
also uses full-record outlier statistics. HUST2 normalizes by maximum
capacity. The warning detector uses trailing windows of the identified state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math
import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from frozen_settings import CITYU_SETTINGS


@dataclass(frozen=True)
class IdentificationConfig:
    """Settings for the capacity-loss model and recursive estimator."""

    scale: float
    rls_lambda: float
    rls_delta: float
    regularization: float
    constraint_multiplier: float
    initial_A: float
    variance_window: int
    qref_mode: str
    warmup_points: int = 10


CONFIGS = {
    "CityU": IdentificationConfig(**CITYU_SETTINGS['identification']),
    "MIT": IdentificationConfig(5000.0, 0.99, 1e3, 0.1, 4.0, 0.1, 30, "first"),
    "HUST1": IdentificationConfig(5000.0, 0.98, 1e3, 0.1, 4.0, 0.1, 30, "maximum"),
    "HUST2": IdentificationConfig(10000.0, 0.99, 1e3, 0.02, 4.0, 1e-4, 30, "maximum"),
    "ISEA": IdentificationConfig(500.0, 0.90, 1e2, 0.1, 1.0, 1e-4, 30, "first"),
    "Tongji": IdentificationConfig(1000.0, 0.90, 1e3, 0.1, 4.0, 0.1, 10, "first"),
}

DATASETS = ("MIT", "HUST1", "HUST2", "ISEA", "Tongji")
EXPECTED_COUNTS = {"CityU": 17, "MIT": 124, "HUST1": 25, "HUST2": 77, "ISEA": 47, "Tongji": 21}
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

PREPROCESSING_NOTES = {
    "CityU": "Measured discharge capacity relative to the manual REF through the first valid 80% crossing.",
    "MIT": "Global 3-sigma repair; full-record UnivariateSpline with s=0.5.",
    "HUST1": "Discard first 11 entries; retain data before first capacity <0.8 Ah.",
    "HUST2": (
        "Discard first 9 dq entries; normalize by full-curve maximum; retain "
        "data before first normalized capacity <0.8. Fit the knee on the full "
        "normalized capacity curve."
    ),
    "ISEA": "Use the released lbl capacity trajectory without additional filtering.",
    "Tongji": "Global 3-sigma repair; divide by the first repaired capacity.",
}


def dataset_configuration(dataset: str) -> dict[str, Any]:
    """Return settings and scope notes for display in a notebook table."""
    if dataset not in CONFIGS:
        raise ValueError(f"Unknown battery dataset {dataset!r}; choose from {DATASETS}.")
    return {
        "dataset": dataset,
        **asdict(CONFIGS[dataset]),
        "expected_records": EXPECTED_COUNTS[dataset],
        "state": "A: effective SEI-associated capacity-loss coefficient",
        "observation_model": "Qref - Q = A*sqrt(i/N0) + B*(i/N0)**1.5",
        "preprocessing": PREPROCESSING_NOTES[dataset],
        "warning_coordinate": "retained-observation index + source cycle origin (Tongji starts at 1)",
        "transition_method": "minimum-SSE split of two independent linear regressions",
        "transition_scope": "complete normalized curve" if dataset == "HUST2" else "retained output curve",
        "upstream_history_only": False,
    }


def _line_sse(prefix: tuple[np.ndarray, ...], start: int, stop: int) -> float:
    """Compute an intercept-inclusive least-squares line SSE on [start, stop)."""
    n = stop - start
    # A line can interpolate one observation exactly.
    if n < 2:
        return 0.0
    px, py, pxx, pyy, pxy = prefix
    sx = px[stop] - px[start]
    sy = py[stop] - py[start]
    sxx = pxx[stop] - pxx[start]
    syy = pyy[stop] - pyy[start]
    sxy = pxy[stop] - pxy[start]
    denominator = sxx - sx * sx / n
    if abs(denominator) < 1e-18:
        return max(0.0, float(syy - sy * sy / n))
    slope = (sxy - sx * sy / n) / denominator
    intercept = (sy - slope * sx) / n
    sse = (
        syy - 2 * intercept * sy - 2 * slope * sxy
        + n * intercept**2 + 2 * intercept * slope * sx + slope**2 * sxx
    )
    # Small negative values can arise from cancellation of prefix sums.
    return max(0.0, float(sse))


def two_segment_linear_transition(clock: np.ndarray, output: np.ndarray) -> int:
    """Return the minimum-SSE split of two independent line fits.

    Candidate indices run from 1 through n-2. A tie takes the first
    candidate. The returned index refers to the supplied observation
    array, and the fit uses finite clock–output pairs.
    """
    x = np.asarray(clock, dtype=float)
    y = np.asarray(output, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("clock and output must be one-dimensional arrays of equal length.")
    valid = np.isfinite(x) & np.isfinite(y)
    original_indices = np.flatnonzero(valid)
    x, y = x[valid], y[valid]
    if len(y) < 3:
        raise ValueError("A two-line reference transition needs at least three finite observations.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("The reference clock must be strictly increasing.")
    prefix = tuple(
        np.r_[0.0, np.cumsum(values)] for values in (x, y, x * x, y * y, x * y)
    )
    best_index, best_sse = 1, float("inf")
    for index in range(1, len(y) - 1):
        sse = _line_sse(prefix, 0, index) + _line_sse(prefix, index, len(y))
        if sse < best_sse:
            best_index, best_sse = index, sse
    return int(original_indices[best_index])


def degradation_model(A: np.ndarray, B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the reduced capacity-loss observation model."""
    x = np.asarray(x, dtype=float)
    return np.asarray(A) * np.sqrt(x) + np.asarray(B) * np.power(x, 1.5)


class ConstrainedRLS:
    """Update two capacity-loss coefficients, with a nonnegative A."""

    def __init__(self, config: IdentificationConfig):
        self.config = config
        self.theta = np.array([config.initial_A, 0.0], dtype=float)
        self.P = np.eye(2, dtype=float) * config.rls_delta

    def update(self, x: float, loss: float) -> np.ndarray:
        """Update A and B using one scaled cycle and one capacity-loss value."""
        phi = np.array([math.sqrt(max(x, 0.0)), max(x, 0.0) ** 1.5], dtype=float)
        pphi = self.P @ phi
        gain = pphi / (self.config.rls_lambda + phi @ pphi)
        self.theta += gain * (loss - phi @ self.theta)
        if self.theta[0] < 0:
            self.theta[0] += (
                self.config.regularization * abs(self.theta[0])
                * self.config.constraint_multiplier
            )
            self.theta[0] = max(self.theta[0], 0.0)
        self.P = (self.P - np.outer(gain, phi @ self.P)) / self.config.rls_lambda
        return self.theta.copy()


def identify_battery_parameters(capacity: np.ndarray, dataset: str) -> dict[str, Any]:
    """Identify A and B on a scaled retained-observation index.

    Initialize the estimator using the warm-up prefix, then update it over
    the complete capacity sequence starting at its first observation.
    """
    config = CONFIGS[dataset]
    capacity = np.asarray(capacity, dtype=float)
    if capacity.ndim != 1 or not len(capacity) or not np.all(np.isfinite(capacity)):
        raise ValueError("Capacity must be a nonempty one-dimensional finite array.")
    qref = float(np.max(capacity)) if config.qref_mode == "maximum" else float(capacity[0])
    loss = qref - capacity
    x = np.arange(len(capacity), dtype=float) / config.scale
    estimator = ConstrainedRLS(config)
    for xx, yy in zip(x[:config.warmup_points], loss[:config.warmup_points]):
        estimator.update(float(xx), float(yy))
    history = np.asarray(
        [estimator.update(float(xx), float(yy)) for xx, yy in zip(x, loss)]
    )
    fitted_loss = degradation_model(history[:, 0], history[:, 1], x)
    return {
        "A": history[:, 0],
        "B": history[:, 1],
        "qref": qref,
        "capacity_loss": loss,
        "scaled_age": x,
        "fitted_loss": fitted_loss,
        "fitted_capacity": qref - fitted_loss,
    }


def _three_sigma(values: np.ndarray, pandas_interpolation: bool = False) -> np.ndarray:
    """Replace observations beyond three full-record standard deviations."""
    y = np.asarray(values, dtype=float).copy()
    lower = np.mean(y) - 3 * np.std(y)
    upper = np.mean(y) + 3 * np.std(y)
    bad = (y < lower) | (y > upper)
    if pandas_interpolation:
        # Tongji uses linear interpolation and edge filling.
        y[bad] = np.nan
        # The normalization step updates this array in place.
        return pd.Series(y).interpolate(method="linear").bfill().ffill().to_numpy(copy=True)
    if np.any(bad):
        index = np.arange(len(y), dtype=float)
        y[bad] = np.interp(index[bad], index[~bad], y[~bad])
    return y


def reconstruct_battery_state(capacity: np.ndarray, dataset: str) -> np.ndarray:
    """Return the identified capacity-loss coefficient A."""
    return identify_battery_parameters(capacity, dataset)["A"]


def fit_diagnostics(capacity: np.ndarray, dataset: str) -> dict[str, Any]:
    """Compute in-sample reconstruction errors using the posterior RLS coefficients."""
    result = identify_battery_parameters(capacity, dataset)
    capacity = np.asarray(capacity, dtype=float)
    residual = capacity - result["fitted_capacity"]
    result.update({
        "residual": residual,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "fit_scope": "in-sample posterior RLS reconstruction",
    })
    return result


def _read_pickle(path: Path) -> Any:
    """Read a released data pickle; use only trusted dataset files."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def _make_record(
    dataset: str,
    cell_id: str,
    capacity: np.ndarray,
    plot_clock: np.ndarray,
    transition_output: np.ndarray | None = None,
    transition_clock: np.ndarray | None = None,
) -> dict[str, Any]:
    """Keep warning indices distinct from plotted physical-cycle coordinates."""
    capacity = np.asarray(capacity, dtype=float)
    plot_clock = np.asarray(plot_clock, dtype=float)
    if transition_output is None:
        transition_output = capacity
    if transition_clock is None:
        transition_clock = plot_clock
    transition_clock = np.asarray(transition_clock, dtype=float)
    tau = two_segment_linear_transition(transition_clock, transition_output)
    parameters = identify_battery_parameters(capacity, dataset)
    if len(plot_clock) > 1 and not np.all(np.diff(plot_clock) == 1):
        raise ValueError(
            f"{dataset}/{cell_id}: source cycle grid is not unit-spaced. "
            "This analysis requires a unit-spaced cycle grid."
        )
    warning_origin = float(plot_clock[0])
    return {
        "id": str(cell_id),
        "dataset": dataset,
        "clock": np.arange(len(capacity), dtype=float) + warning_origin,
        "retained_index": np.arange(len(capacity), dtype=float),
        "plot_clock": plot_clock,
        "state": parameters["A"],
        "output": capacity,
        "tau": float(tau) + warning_origin,
        "transition_index": int(tau),
        "transition_plot_clock": float(transition_clock[tau]),
        "n_engineering": len(capacity),
        "basew": CONFIGS[dataset].variance_window,
        "origin": warning_origin,
        "transition_fit_clock": transition_clock,
        "transition_fit_output": np.asarray(transition_output, dtype=float),
        "B": parameters["B"],
        "state_B": parameters["B"],
        "qref": parameters["qref"],
        "model_output": parameters["fitted_capacity"],
        "fitted_capacity": parameters["fitted_capacity"],
        "capacity_loss": parameters["capacity_loss"],
        "scaled_age": parameters["scaled_age"],
        "metadata": {
            "preprocessing": PREPROCESSING_NOTES[dataset],
            "transition_source_points": len(transition_output),
            "retained_points": len(capacity),
            "transition_within_retained_record": tau < len(capacity),
            "warning_coordinate": "retained-observation index + source cycle origin",
            "plot_coordinate": "CSV cycle number" if dataset == "Tongji" else "retained cycle index",
            "warmup_points": min(CONFIGS[dataset].warmup_points, len(capacity)),
            "upstream_history_only": False,
        },
    }


def _dataset_directory(root: Path, dataset: str) -> Path:
    """Find the battery dataset folder in a supported release layout."""
    original_names = {
        "MIT": "MIT_data", "HUST1": "HUST1_data", "HUST2": "HUST2_data",
        "ISEA": "ISEA_data", "Tongji": "Tongji_NCA_data",
    }
    candidates = [root / dataset, root / original_names[dataset]]
    if root.name in (dataset, original_names[dataset]):
        candidates.append(root)
    for directory in candidates:
        if directory.is_dir():
            return directory
    raise FileNotFoundError(
        f"Missing {dataset} input folder. Expected {root / dataset}. "
        "Supply the parent of the dataset folders as data_root."
    )


def _load_mit(root: Path) -> list[dict[str, Any]]:
    labels = {}
    for filename in ("label_train.pkl", "label_test.pkl", "label_sec.pkl"):
        labels.update(_read_pickle(root / filename))
    records = []
    for cell_id, raw in sorted(labels.items()):
        capacity = _three_sigma(np.asarray(raw, dtype=float))
        clock = np.arange(len(capacity), dtype=float)
        capacity = np.asarray(UnivariateSpline(clock, capacity, s=0.5)(clock), dtype=float)
        records.append(_make_record("MIT", str(cell_id), capacity, clock))
    return records


def _load_hust1(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(p for p in root.glob("*.xlsx") if not p.name.startswith((".", "~$"))):
        full = pd.read_excel(path)["Capacity (Ah)"].to_numpy(dtype=float)[11:]
        below = np.flatnonzero(full < 0.8)
        if not len(below):
            raise ValueError(f"HUST1 record has no 0.8 Ah crossing: {path.name}")
        eol = int(below[0])
        capacity = full[:eol]
        records.append(_make_record("HUST1", path.stem, capacity, np.arange(eol, dtype=float)))
    return records


def _load_hust2(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(p for p in root.glob("*.pkl") if not p.name.startswith(".")):
        raw = _read_pickle(path)[path.stem]["dq"]
        # Read each sheet in workbook order and skip nine capacity entries.
        dq = np.asarray([raw[key] for key in list(raw)[9:]], dtype=float)
        full = dq / np.max(dq)
        below = np.flatnonzero(full < 0.8)
        if not len(below):
            raise ValueError(f"HUST2 record has no normalized 0.8 crossing: {path.name}")
        eol = int(below[0])
        clock = np.arange(len(full), dtype=float)
        # Fit the HUST2 knee on the complete normalized curve scaled by 1190.
        records.append(_make_record(
            "HUST2", path.stem, full[:eol], clock[:eol], full * 1190.0, clock
        ))
    return records


def _load_isea(root: Path) -> list[dict[str, Any]]:
    labels = _read_pickle(root / "nmc_loader.pkl")
    records = []
    for cell_id in sorted(labels):
        capacity = np.asarray(labels[cell_id]["lbl"], dtype=float)
        clock = np.arange(len(capacity), dtype=float)
        records.append(_make_record("ISEA", str(cell_id), capacity, clock))
    return records


def _load_tongji(root: Path) -> list[dict[str, Any]]:
    records = []
    # The capacity CSVs are in the dataset folder or its knee subfolder.
    curves = root / "knee" if (root / "knee").is_dir() else root
    for path in sorted(p for p in curves.glob("*.csv") if not p.name.startswith(".")):
        frame = pd.read_csv(path)
        clock = frame["cycle number"].to_numpy(dtype=float)
        capacity = _three_sigma(frame["Q discharge/mA.h"].to_numpy(dtype=float), True)
        capacity /= capacity[0]
        records.append(_make_record(
            "Tongji", path.stem.replace("_Q_cycle", ""), capacity, clock
        ))
    return records


LOADERS = {
    "MIT": _load_mit,
    "HUST1": _load_hust1,
    "HUST2": _load_hust2,
    "ISEA": _load_isea,
    "Tongji": _load_tongji,
}


def load_battery_records(
    dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT
) -> list[dict[str, Any]]:
    """Prepare capacity, identify A and fit a knee for each battery.

    The result includes the source cycle coordinate, the observation
    index used by the detector and the curve used to locate the knee.
    HUST2 locates the knee on its full normalized capacity curve.
    Records are returned in specimen-ID order.
    """
    dataset_configuration(dataset)
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Battery data directory is not accessible: {root}")
    records = sorted(LOADERS[dataset](_dataset_directory(root, dataset)),
                     key=lambda record: record["id"])
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"Duplicate cell IDs in {dataset}.")
    if len(records) != EXPECTED_COUNTS[dataset]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS[dataset]} {dataset} records, found {len(records)}. "
            "Check the dataset folder for the expected capacity files."
        )
    return records
