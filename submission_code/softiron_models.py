"""Fit and load the nonlinear compliance state for soft magnetic iron.

The C2 series uses one-second blocks and a force reference from the first
30 qualifying blocks. The displacement comparator and transition fit
use the full-record loading interval.
"""

from pathlib import Path
import json
import re
import zipfile

import numpy as np
import pandas as pd


DEFAULT_RAW_ROOT = Path(__file__).resolve().parent.parent / "data/Soft_Magnetic_Iron"
FREQUENCY_HZ = 20.0
COEFFICIENT_NAMES = ("C1", "Cq", "C2", "C3", "scaled_gram_condition")
FEATURE_COLUMNS = (
    "second", "t_first", "t_last", "count", "F_min", "F_max", "u_min",
    "u_max", "B_min", "B_max", "F_mean", "u_mean", "B_mean", "F_var",
    "u_var", "B_var", "Fu_cov", "F_cos", "F_sin", "u_cos", "u_sin",
    "B_cos", "B_sin",
)
_ROW = re.compile(rb"<row\b[^>]*>.*?</row>", re.S)
_CELL = re.compile(
    rb'<c\b[^>]*\br="([A-Z]+)\d+"[^>]*>'
    rb'(?:<f\b[^>]*?/>|<f\b[^>]*>.*?</f>)?'
    rb'<v>([-+.0-9eE]+)</v></c>'
)


def _segment_line(x, y):
    """Fit one unconstrained OLS line, including the one-point segment case."""
    dx, dy = x - x.mean(), y - y.mean()
    denominator = dx @ dx
    slope = float(dx @ dy / denominator) if denominator > 0 else 0.0
    intercept = float(y.mean() - slope * x.mean())
    residual = dy - slope * dx
    return slope, intercept, float(residual @ residual)


def two_segment_linear_transition(clock, output):
    """Fit a two-line reference to mean displacement.

    For each split k from 1 through n-2, the fitted segments use [:k] and
    [k:]. The reference time is clock[k] at the minimum total squared error.
    """
    t = np.asarray(clock, dtype=float)
    y = np.asarray(output, dtype=float)
    if t.ndim != 1 or y.shape != t.shape:
        raise ValueError("clock and output must be equal-length one-dimensional arrays")
    if len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(y).all():
        return {"status": "invalid_input"}
    x = np.asarray(t, dtype=np.longdouble)
    z = np.asarray(y, dtype=np.longdouble)
    x_scale, y_scale = max(float(np.ptp(x)), 1.0), max(float(np.ptp(z)), 1e-12)
    x, z = (x - x[0]) / x_scale, (z - z.mean()) / y_scale

    def side(a, b):
        n = np.arange(1, len(a) + 1, dtype=np.longdouble)
        sx, sy = np.cumsum(a), np.cumsum(b)
        xx = np.cumsum(a * a) - sx * sx / n
        yy = np.cumsum(b * b) - sy * sy / n
        xy = np.cumsum(a * b) - sx * sy / n
        explained = np.divide(xy * xy, xx, out=np.zeros_like(xx), where=xx > 0)
        sse = np.maximum(yy - explained, 0)
        sse[0] = 0
        return sse

    splits = np.arange(1, len(t) - 1)
    left, right = side(x, z), side(x[::-1], z[::-1])[::-1]
    errors = np.asarray((left[splits - 1] + right[splits]) * y_scale**2, float)
    tolerance = max(float(np.max(errors)), 1e-30) * 1e-12
    contenders = splits[errors <= np.min(errors) + tolerance]
    direct = [
        _segment_line(t[:k], y[:k])[2] + _segment_line(t[k:], y[k:])[2]
        for k in contenders
    ]
    # Take the earliest split in an exact tie.
    k = int(contenders[np.argmin(direct)])
    fitted_left, fitted_right = _segment_line(t[:k], y[:k]), _segment_line(t[k:], y[k:])
    sse = fitted_left[2] + fitted_right[2]
    return {
        "status": "candidate", "tau": float(t[k]), "tau_s": float(t[k]),
        "split_index": k, "fit_points": len(t), "sse": sse,
        "left_slope": fitted_left[0], "right_slope": fitted_right[0],
        "left_intercept": fitted_left[1], "right_intercept": fitted_right[1],
        "fit_improvement": 1 - sse / max(_segment_line(t, y)[2], 1e-30),
        "later_slope_greater": bool(fitted_right[0] > fitted_left[0]),
        "note": "All splits searched; no scientific acceptance filter",
    }


def causal_c2_series(features, coefficient_clock, c2, stop_s=None):
    """Place accepted C2 estimates on the full one-second clock.

    A block [s,s+1) becomes available at s+1. Missing values remain
    on the grid. The force reference is the median amplitude of the
    first 30 qualifying blocks.
    """
    a = np.asarray(features, dtype=float)
    coefficient_clock, c2 = np.asarray(coefficient_clock), np.asarray(c2)
    if stop_s is not None:
        a = a[a[:, 0] + 1 <= stop_s]
        keep = coefficient_clock <= stop_s
        coefficient_clock, c2 = coefficient_clock[keep], c2[keep]
    if len(a) == 0:
        raise ValueError("No completed one-second feature blocks")
    t = a[:, 0] + 1
    force, displacement = a[:, 17] + 1j * a[:, 18], a[:, 19] + 1j * a[:, 20]
    amplitude = np.abs(force)
    force_fit = amplitude**2 / (2 * np.maximum(a[:, 13], 1e-30))
    driven = (amplitude > 1) & (force_fit > 0.9) & (a[:, 3] >= 120)
    driven_indices = np.flatnonzero(driven)
    clock = np.arange(int(t[0]), int(t[-1]) + 1, dtype=float)
    state, observed_output = np.full(len(clock), np.nan), np.full(len(clock), np.nan)
    positions = (t - clock[0]).astype(int)
    observed_output[positions] = a[:, 11]
    origin = float(t[driven_indices[0]]) if len(driven_indices) else None
    reference, ready = None, None
    if len(driven_indices) >= 30:
        first = driven_indices[:30]
        reference, ready = float(np.median(amplitude[first])), float(t[first[-1]])
        displacement_fit = np.abs(displacement)**2 / (2 * np.maximum(a[:, 14], 1e-30))
        with np.errstate(divide="ignore", invalid="ignore"):
            compliance = displacement / np.where(amplitude > 0, force, np.nan)
        quality = (
            driven & (t >= ready) & (amplitude > 0.9 * reference)
            & (amplitude < 1.1 * reference) & (displacement_fit > 0.95)
            & (displacement_fit < 1.05) & (compliance.real > 0)
        )
        coefficient_positions = (coefficient_clock - clock[0]).astype(int)
        keep = (coefficient_positions >= 0) & (coefficient_positions < len(clock))
        state[coefficient_positions[keep]] = c2[keep]
        gate = np.zeros(len(clock), bool)
        gate[positions] = quality
        state[~gate] = np.nan
    return {
        "clock": clock, "state": state, "observed_output": observed_output,
        "origin": origin, "force_reference": reference,
        "force_reference_ready_s": ready,
    }


def historical_output_series(features):
    """Prepare mean displacement for the comparator and transition fit.

    The loading interval uses a full-record median force amplitude and
    is trimmed by five seconds at each end. The transition fit uses
    the active blocks before the displacement-quality check.
    """
    a = np.asarray(features, float)
    t = a[:, 0] + 1
    force, displacement = a[:, 17] + 1j * a[:, 18], a[:, 19] + 1j * a[:, 20]
    amplitude = np.abs(force)
    force_fit = amplitude**2 / (2 * np.maximum(a[:, 13], 1e-30))
    driven = (amplitude > 1) & (force_fit > 0.9) & (a[:, 3] >= 120)
    if not driven.any():
        raise ValueError("No driven blocks in the output interval")
    target = np.median(amplitude[driven])
    active = (
        (amplitude > 0.9 * target) & (amplitude < 1.1 * target)
        & (a[:, 3] >= 120) & (force_fit > 0.9)
    )
    active_indices = np.flatnonzero(active)
    start, end = int(t[active_indices[0]] + 5), int(t[active_indices[-1]] - 5)
    use = active & (t >= start) & (t <= end)
    with np.errstate(divide="ignore", invalid="ignore"):
        compliance = displacement / np.where(amplitude > 0, force, np.nan)
    displacement_fit = np.abs(displacement)**2 / (2 * np.maximum(a[:, 14], 1e-30))
    quality = use & (displacement_fit > 0.95) & (displacement_fit < 1.05) & (compliance.real > 0)
    clock = np.arange(start, end + 1, dtype=float)
    output = np.full(len(clock), np.nan)
    keep = (t >= start) & (t <= end)
    output[t[keep].astype(int) - start] = np.where(quality, a[:, 11], np.nan)[keep]
    return {
        "output_clock": clock, "output": output,
        "transition_clock": t[use], "transition_output": a[use, 11],
        "archived_force_reference": float(target),
    }


def load_softiron_records(audit_root, recompute_transition=False, transitions_path=None):
    """Assemble the 110 fatigue histories from one-second blocks and fits.

    The inputs include the specimen manifest and displacement references.
    The optional fit recomputes the reference for verification.
    """
    root = Path(audit_root)
    manifest = pd.read_csv(root / "specimen_split/split_manifest.csv")
    manifest = manifest.loc[manifest.material.eq("Soft Magnetic Iron")].sort_values("id")
    # Use the reference table supplied with the prepared inputs.
    local_transitions = Path(__file__).resolve().parent / "frozen/fatigue_transitions.csv"
    if transitions_path is None:
        transitions_path = local_transitions if local_transitions.exists() else root / "transition_reference/transition_comparison.csv"
    labels = pd.read_csv(transitions_path)
    labels = labels.loc[labels.material.eq("Soft Magnetic Iron")].set_index("id")
    if len(manifest) != 110 or len(labels) != 110:
        raise ValueError("Expected the frozen 110-specimen soft-iron cohort")
    records = []
    for entry in manifest.itertuples():
        sid = int(entry.id)
        with np.load(root / "features" / f"{sid:03d}.npz") as source:
            features = source["data"]
        coefficient_path = root / "pretransition_state_screen/c2_cohort" / f"coefficients_{sid:03d}.npz"
        with np.load(coefficient_path) as source:
            durations = source["identification_seconds"]
            if int(durations[0]) != 1:
                raise ValueError(f"Specimen {sid}: expected the frozen 1-second C2 fit")
            prepared = causal_c2_series(features, source["t"], source["coefficients"][:, 0, 2])
        archived = historical_output_series(features)
        label = labels.loc[sid].to_dict()
        tau = float(label["new_tau_s"])
        if recompute_transition:
            fit = two_segment_linear_transition(archived["transition_clock"], archived["transition_output"])
            if fit.get("tau") != tau:
                raise AssertionError((sid, fit.get("tau"), tau))
        n_engineering = int(label["loading_end_s"] - label["loading_start_s"] + 1)
        records.append({
            "id": str(sid), "dataset": "Soft_Magnetic_Iron", "role": entry.role,
            "clock": prepared["clock"], "state": prepared["state"],
            "output_clock": archived["output_clock"], "output": archived["output"],
            "transition_fit_clock": archived["transition_clock"],
            "transition_fit_output": archived["transition_output"],
            "tau": tau, "n_engineering": n_engineering,
            "label": label, "basew": 10, "origin": prepared["origin"],
            "force_reference": prepared["force_reference"],
            "force_reference_ready_s": prepared["force_reference_ready_s"],
            "coefficient_cache": str(coefficient_path), "state_label": "C2 (mm/kN^2)",
            "output_label": "Mean displacement (mm)", "clock_label": "Recorded time (s)",
            "direction": "up",
        })
    return records


def _numeric_blocks(archive, member, specimen_id, chunk_bytes=4 * 1024 * 1024):
    """Stream the supplied workbook's numerical channels, retaining timestamps."""
    pending = b""
    with archive.open(member) as stream:
        while True:
            more = stream.read(chunk_bytes)
            pending += more
            last = pending.rfind(b"</row>")
            if last < 0:
                if not more:
                    break
                continue
            xml, pending = pending[:last + 6], pending[last + 6:]
            if b't="s"' in xml:
                xml = b"".join(row for row in _ROW.findall(xml) if b't="s"' not in row)
            cells = _CELL.findall(xml)
            if not cells:
                if not more:
                    break
                continue
            if specimen_id == 3 and member != "xl/worksheets/sheet1.xml":
                # A=time, B=derived cycles, C=force, D=displacement, E=voltage,
                # F=source-provided calibrated magnetic signal. No new calibration.
                columns, width, take = b"ABCDEF", 6, [0, 2, 3, 5]
            else:
                # For specimen 002, read the measured channels in columns A–D.
                extra = {cell[0] for cell in cells} - {b"A", b"B", b"C", b"D"}
                if extra and not (specimen_id == 2 and extra == {b"E"}):
                    raise ValueError(f"Unexpected workbook columns: {member}, {extra}")
                cells = [cell for cell in cells if cell[0] in (b"A", b"B", b"C", b"D")]
                columns, width, take = b"ABCD", 4, [0, 1, 2, 3]
            if len(cells) % width or b"".join(c[0] for c in cells) != columns * (len(cells) // width):
                raise ValueError(f"Missing, reordered or nonnumeric workbook cells: {member}")
            values = np.fromstring(b" ".join(c[1] for c in cells).decode("ascii"), sep=" ")
            data = values.reshape(-1, width)[:, take]
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite raw measurements: {member}")
            yield data
            if not more:
                break


def read_raw_interval(workbook_path, specimen_id, start_s=None, end_s=None):
    """Read raw time, force, displacement and magnetic channels on [start_s,end_s)."""
    blocks, previous = [], -np.inf
    with zipfile.ZipFile(workbook_path) as archive:
        sheets = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
            key=lambda name: int(re.search(r"sheet(\d+)", name)[1]),
        )
        finished = False
        for member in sheets:
            for data in _numeric_blocks(archive, member, int(specimen_id)):
                if data[0, 0] <= previous or np.any(np.diff(data[:, 0]) <= 0):
                    raise ValueError("Raw measurement timestamps must strictly increase")
                previous = data[-1, 0]
                keep = np.ones(len(data), bool)
                if start_s is not None:
                    keep &= data[:, 0] >= start_s
                if end_s is not None:
                    keep &= data[:, 0] < end_s
                if keep.any():
                    blocks.append(data[keep])
                if end_s is not None and data[-1, 0] >= end_s:
                    finished = True
                    break
            if finished:
                break
    return np.vstack(blocks) if blocks else np.empty((0, 4))


def _block_moments(raw, starts, ends):
    """Residualized per-second normal equations for the physical coefficients."""
    groups = []
    lengths = ends - starts
    for length in np.unique(lengths):
        if length < 120:
            continue
        selected = np.flatnonzero(lengths == length)
        block = raw[starts[selected, None] + np.arange(length)[None, :]]
        t = block[:, :, 0]
        centered_time = t - t.mean(axis=1, keepdims=True)
        phase = 2 * np.pi * FREQUENCY_HZ * (t - np.floor(t[:, 0, None]))
        harmonic = np.stack(
            [np.ones_like(t), centered_time]
            + [v for order in (1, 2, 3) for v in (np.cos(order * phase), np.sin(order * phase))],
            axis=2,
        )
        transposed = harmonic.swapaxes(1, 2)
        harmonic_beta = np.linalg.solve(transposed @ harmonic, transposed @ block[:, :, 1:3])
        force_real, force_imag = harmonic_beta[:, 2, 0], harmonic_beta[:, 3, 0]
        amplitude = np.hypot(force_real, force_imag)
        storage = (harmonic_beta[:, 2, 1] * force_real + harmonic_beta[:, 3, 1] * force_imag) / np.maximum(amplitude**2, 1e-20)
        p = block[:, :, 1] - block[:, :, 1].mean(axis=1, keepdims=True)
        quadrature = force_real[:, None] * np.sin(phase) - force_imag[:, None] * np.cos(phase)
        predictors = np.stack([p, quadrature, p**2, p**3], axis=2)
        nuisance = harmonic[:, :, :2]
        nuisance_transposed = nuisance.swapaxes(1, 2)
        joint = np.concatenate([predictors, block[:, :, 2, None]], axis=2)
        residualized = joint - nuisance @ np.linalg.solve(nuisance_transposed @ nuisance, nuisance_transposed @ joint)
        predictors, response = residualized[:, :, :4], residualized[:, :, 4]
        gram = predictors.swapaxes(1, 2) @ predictors
        rhs = (predictors.swapaxes(1, 2) @ response[:, :, None])[:, :, 0]
        valid = (amplitude > 1) & (storage > 0) & (np.max(np.diff(t, axis=1), axis=1) <= 0.02)
        gram[~valid], rhs[~valid] = np.nan, np.nan
        groups.append(np.c_[np.floor(t[:, 0]) + 1, gram.reshape(-1, 16), rhs, p.min(axis=1), p.max(axis=1)])
    return np.vstack(groups) if groups else np.empty((0, 23))


def _identify_moments(moments, identification_seconds=1):
    """Fit C1,Cq,C2,C3 with a positive tangent across 51 observed-load points."""
    if not len(moments):
        return np.array([]), np.empty((0, 5))
    moments = moments[np.argsort(moments[:, 0])]
    clock = np.arange(int(moments[0, 0]), int(moments[-1, 0]) + 1)
    data = np.full((len(clock), 22), np.nan)
    data[(moments[:, 0] - clock[0]).astype(int)] = moments[:, 1:]
    sums = pd.DataFrame(data[:, :20]).rolling(identification_seconds).sum().to_numpy()
    gram, rhs = sums[:, :16].reshape(-1, 4, 4), sums[:, 16:20]
    lo = pd.Series(data[:, 20]).rolling(identification_seconds).min().to_numpy()
    hi = pd.Series(data[:, 21]).rolling(identification_seconds).max().to_numpy()
    valid = np.isfinite(sums).all(axis=1)
    coefficients, condition = np.full((len(clock), 4), np.nan), np.full(len(clock), np.nan)
    scales = np.sqrt(np.diagonal(gram[valid], axis1=1, axis2=2))
    normalized = gram[valid] / scales[:, :, None] / scales[:, None, :]
    beta = np.linalg.solve(normalized, (rhs[valid] / scales)[:, :, None])[:, :, 0]
    coefficients[valid], condition[valid] = beta / scales, np.linalg.cond(normalized)
    p = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, 51)
    tangent = coefficients[:, 0, None] + 2 * coefficients[:, 2, None] * p + 3 * coefficients[:, 3, None] * p**2
    coefficients[~(np.min(tangent, axis=1) > 0)] = np.nan
    return clock, np.c_[coefficients, condition]


def fit_c2_from_raw(raw, identification_seconds=1, completed_until=None):
    """Rebuild the measured-force nonlinear response fit on supplied raw rows.

    u = intercept + drift*centered_time + C1*p + Cq*p90 + C2*p^2 + C3*p^3.
    Per-second intercept and drift are nuisance terms projected out jointly.
    Identification uses only completed blocks ending at or before the supplied
    completed_until. Regressors use powers of the measured force. C2 has units
    mm/kN^2 for the supplied force and displacement channels.
    """
    a = np.asarray(raw, dtype=float)
    if a.ndim != 2 or a.shape[1] < 3 or not np.isfinite(a[:, :3]).all():
        raise ValueError("raw must contain finite time, force and displacement columns")
    if len(a) == 0:
        return {"clock": np.array([]), "coefficients": np.empty((0, 5)), "names": COEFFICIENT_NAMES}
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError("Raw measurement timestamps must strictly increase")
    if not isinstance(identification_seconds, int) or identification_seconds < 1:
        raise ValueError("identification_seconds must be a positive integer")
    # Without an explicit arrival-time cutoff the final bin may be incomplete.
    end = float(completed_until) if completed_until is not None else float(np.floor(a[-1, 0]))
    bins = np.floor(a[:, 0])
    a = a[bins + 1 <= end]
    if not len(a):
        return {"clock": np.array([]), "coefficients": np.empty((0, 5)), "names": COEFFICIENT_NAMES}
    bins = np.floor(a[:, 0])
    cuts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(a)]
    moments = _block_moments(a, cuts[:-1], cuts[1:])
    clock, coefficients = _identify_moments(moments, identification_seconds)
    return {"clock": clock, "coefficients": coefficients, "names": COEFFICIENT_NAMES}


def verify_raw_c2_subset(audit_root, specimen_id=119, end_s=120, raw_root=DEFAULT_RAW_ROOT):
    """Compare the one-second response fits for a specified workbook interval."""
    root, sid = Path(audit_root), int(specimen_id)
    metadata = json.loads((root / "features" / f"{sid:03d}.json").read_text())
    raw_path = Path(raw_root) / metadata["file"]
    raw = read_raw_interval(raw_path, sid, end_s=end_s)
    rebuilt = fit_c2_from_raw(raw, identification_seconds=1, completed_until=end_s)
    with np.load(root / "pretransition_state_screen/c2_cohort" / f"coefficients_{sid:03d}.npz") as source:
        positions = (rebuilt["clock"] - source["t"][0]).astype(int)
        reference = source["coefficients"][positions, 0, :]
    expected, actual = reference[:, 2], rebuilt["coefficients"][:, 2]
    finite = np.isfinite(expected) & np.isfinite(actual)
    match = np.array_equal(np.isnan(expected), np.isnan(actual)) and np.allclose(expected[finite], actual[finite], rtol=1e-6, atol=1e-12)
    return {
        "id": str(sid), "raw_path": str(raw_path), "raw_rows": len(raw),
        "completed_blocks": len(actual), "finite_C2_blocks": int(finite.sum()),
        "C2_cache_match": bool(match),
        "maximum_C2_absolute_difference": float(np.max(np.abs(expected[finite] - actual[finite]))) if finite.any() else np.nan,
        "clock": rebuilt["clock"], "rebuilt_coefficients": rebuilt["coefficients"],
        "cached_coefficients": reference, "raw": raw,
    }
