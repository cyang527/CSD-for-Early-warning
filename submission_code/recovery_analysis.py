"""Battery state-variance contrasts and quadratic-trend recovery diagnostics."""

import numpy as np
import pandas as pd

from battery_models import CONFIGS
from manuscript_replay import BATCHES, FATIGUE, records

BATTERY_BATCHES = tuple(batch for batch in BATCHES if batch != FATIGUE)
RECOVERY_WINDOW = 60
FIGURE3_CELL = ("HUST1", "2-4")


def comparison_intervals(clock, knee):
    """Earlier and near-knee intervals on the retained observation clock."""
    span = clock[-1] - clock[0]
    return {"early": (knee - .30 * span, knee - .20 * span),
            "near": (knee - .05 * span, knee)}


def variance_components(clock, state, window, endpoints):
    """Separate sample variance into linear trend and residual contributions."""
    rows = []
    for end in endpoints:
        if end < window - 1:
            continue
        times = clock[end - window + 1:end + 1]
        values = state[end - window + 1:end + 1]
        if not np.isfinite(values).all():
            continue
        x = times - times.mean()
        if not (x @ x > 0):
            continue
        y = values - values.mean()
        trend = x * (x @ y) / (x @ x)
        residual = y - trend
        rows.append(((y @ y) / (window - 1), (trend @ trend) / (window - 1),
                     (residual @ residual) / (window - 1)))
    return np.asarray(rows).reshape(-1, 3)


def recovery_trace(clock, state, endpoints, window=RECOVERY_WINDOW,
                   innovation_diagnostics=False):
    """Fit a quadratic time trend and an AR(1) coefficient in each window."""
    clock, state = np.asarray(clock, float), np.asarray(state, float)
    endpoints = np.asarray(endpoints, int)
    trend = np.linspace(-.5, .5, window)[1:]
    regressors = np.column_stack([np.ones(window - 1), trend, trend**2])
    rows = []
    for end in endpoints:
        phi, kappa, noise, lag1 = np.nan, np.nan, np.nan, np.nan
        status = "insufficient_history"
        if end >= window - 1:
            sl = slice(end - window + 1, end + 1)
            step = np.diff(clock[sl])
            values = state[sl]
            if np.any(step <= 0) or not np.allclose(step, step[0], rtol=1e-10, atol=1e-12):
                status = "irregular_clock"
            elif not np.isfinite(values).all() or values.std() == 0:
                status = "invalid_state"
            else:
                normalized = (values - values.mean()) / values.std()
                design = np.column_stack([regressors, normalized[:-1]])
                status = "rank_deficient"
                if np.linalg.matrix_rank(design) == design.shape[1]:
                    coefficients = np.linalg.lstsq(design, normalized[1:], rcond=None)[0]
                    phi = coefficients[-1]
                    if phi <= 0:
                        status = "phi_nonpositive"
                    elif phi >= 1:
                        status = "phi_ge_one"
                    else:
                        status = "valid"
                        kappa = -np.log(phi) / step[0]
                        if innovation_diagnostics:
                            innovation = (normalized[1:] - design @ coefficients) * values.std()
                            noise = 2 * kappa * np.var(innovation, ddof=1) / (1 - phi**2)
                            if np.std(innovation[:-1]) > 0 and np.std(innovation[1:]) > 0:
                                lag1 = np.corrcoef(innovation[:-1], innovation[1:])[0, 1]
        rows.append((clock[end], phi, kappa, noise, lag1, status))
    return pd.DataFrame(rows, columns=["clock", "phi", "kappa", "sigma_sq",
                                       "innovation_lag1", "status"])


def _ratio(near, early):
    return near / early if np.isfinite(near) and np.isfinite(early) and early > 0 else np.nan


def battery_diagnostics(histories_by_batch=None):
    """Return capacity-state records and the Figure 2 and Figure 3 cohort tables."""
    cells, variance_rows, recovery_rows = {}, [], []
    for batch in BATTERY_BATCHES:
        if histories_by_batch is None:
            batch_records = records(batch)
        else:
            batch_records = [value["record"] for value in histories_by_batch[batch].values()]
        for cell in batch_records:
            sid = str(cell["id"])
            cells[(batch, sid)] = cell
            clock, state, knee = np.asarray(cell["clock"], float), np.asarray(cell["state"], float), float(cell["tau"])
            internal_knee = clock[0] < knee < clock[-1]
            variance_window = CONFIGS[batch].variance_window
            identity = dict(batch=batch, dataset="HUST" if batch.startswith("HUST") else batch,
                            id=sid, n=len(clock), tau=knee)
            var = dict(identity, internal_knee=internal_knee, variance_window=variance_window)
            rec = dict(identity, window=RECOVERY_WINDOW)
            for period, (start, stop) in comparison_intervals(clock, knee).items():
                endpoints = np.flatnonzero((clock >= start) & (clock < stop))
                components = variance_components(clock, state, variance_window, endpoints)
                means = components.mean(axis=0) if len(components) else [np.nan] * 3
                var.update({period + "_" + k: v for k, v in zip(("raw", "trend", "residual"), means)})
                var[period + "_raw_median"] = np.median(components[:, 0]) if len(components) else np.nan
                var[period + "_count"] = len(components)
                var[period + "_trend_share"] = _ratio(means[1], means[0])
                trace = recovery_trace(clock, state, endpoints)
                count = int(trace.kappa.count())
                rec[period + "_count"] = count
                rec[period + "_endpoints"] = len(endpoints)
                rec[period + "_coverage"] = count / len(endpoints) if len(endpoints) else 0.
                rec[period + "_kappa"] = float(trace.kappa.median()) if count else np.nan
            var["variance_evaluable"] = bool(internal_knee and min(var["early_count"], var["near_count"]) >= 3)
            for quantity in ("raw", "raw_median", "residual"):
                var[quantity + "_ratio"] = (_ratio(var["near_" + quantity], var["early_" + quantity])
                                            if var["variance_evaluable"] else np.nan)
            rec["residual_ratio"] = var["residual_ratio"]
            rec["raw_ratio"] = var["raw_ratio"]
            rec["evaluable"] = bool(var["variance_evaluable"] and min(rec["early_count"], rec["near_count"]) >= 3)
            rec["kappa_ratio"] = _ratio(rec["near_kappa"], rec["early_kappa"]) if rec["evaluable"] else np.nan
            rec["recovery_decrease"] = bool(rec["evaluable"] and rec["kappa_ratio"] < 1)
            rec["residual_increase"] = bool(rec["evaluable"] and rec["residual_ratio"] > 1)
            rec["joint_direction"] = rec["recovery_decrease"] and rec["residual_increase"]
            rec["coverage50"] = bool(rec["evaluable"] and min(rec["early_coverage"], rec["near_coverage"]) >= .5)
            rec["joint_coverage50"] = rec["coverage50"] and rec["joint_direction"]
            variance_rows.append(var)
            recovery_rows.append(rec)
    return cells, pd.DataFrame(variance_rows), pd.DataFrame(recovery_rows)
