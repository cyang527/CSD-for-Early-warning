"""Analysis tables and specimen traces for the warning and fatigue figures."""

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from battery_models import CONFIGS
from ews_core import confirmed_alarm, persistence_detector, raw_variance, score_alarm
from manuscript_replay import FATIGUE, settings
from recovery_analysis import comparison_intervals, variance_components

DATASETS = ("MIT", "HUST", "CityU", "ISEA", "Tongji")
GROUPS = ("G1", "G2", "G3", "G4")
METHODS = ("Proposed", "MR", "PM", "IQRM", "TSM")
RULES = ("Level only", "Growth only", "Joint rule")
MATRIX_ORDER = (
    Path(__file__).resolve().parent.parent / "submission_inputs" / "publication_figures"
    / "fig456" / "matrix_order.csv"
)


def _raw_id(batch, specimen_id):
    return specimen_id.removeprefix(batch + "__") if batch.startswith("HUST") else specimen_id


def _history(histories, row):
    return histories[row.batch][_raw_id(row.batch, row.id)]


def _persistence_parts(variance, config):
    length = int(config["d"])
    series = pd.Series(variance)
    recent = series.rolling(length, min_periods=ceil(.9 * length))
    median = recent.median().to_numpy()
    if config["h"] == 1:
        short = np.full(len(series), np.nan)
        persistence = recent.quantile(.1).to_numpy()
    else:
        short_length = max(2, ceil(length / 3))
        short_series = series.rolling(short_length, min_periods=ceil(.8 * short_length)).median()
        short = short_series.to_numpy()
        persistence = short_series.rolling(length).min().to_numpy()
    return median, short, np.minimum(persistence, median / config["r"])


def warning_case(batch, specimen_id, histories):
    """Return the plotted detector inputs and stage overlays for a figure example."""
    config, _, thresholds = settings(batch)
    threshold = thresholds["G4"]
    history = histories[batch][specimen_id]
    record, annotation = history["record"], history["annotation"]
    clock = record["clock"]
    trace = persistence_detector(record["state"], config)
    alarm, index, _ = confirmed_alarm(clock, trace["score"], threshold)
    median, short, bound = _persistence_parts(trace["variance"], config)
    reference = trace["reference"]
    case_trace = dict(
        clock=clock, transition_s=record["tau"], alarm_s=alarm, alarm_index=index,
        threshold=threshold, episode_start=[i["start_s"] for i in annotation["intervals"]],
        episode_end=[i["end_s"] for i in annotation["intervals"]],
        persistence_bound=bound, short_median=short,
        recent_median_relative_to_reference=median / reference,
        persistence_relative_to_reference=bound / reference, **trace,
    )
    return dict(dataset=batch, id=specimen_id, config_parameters=config, record=case_trace)


def condition_outcomes(batteries, histories):
    """Apply the two individual conditions and the joint rule to identical histories."""
    rows = []
    configurations = {batch: settings(batch) for batch in batteries.batch.unique()}
    for row in batteries.itertuples(index=False):
        config, _, thresholds = configurations[row.batch]
        history = _history(histories, row)
        record, annotation = history["record"], history["annotation"]
        trace = persistence_detector(record["state"], config)
        _, _, bound = _persistence_parts(trace["variance"], config)
        ready = np.isfinite(trace["score"])
        scores = (
            (np.where(ready, bound / trace["reference"], np.nan), thresholds["G4"]),
            (np.where(ready, trace["gain"] >= config["growth"], np.nan), 0),
            (trace["score"], thresholds["G4"]),
        )
        for rule, (score, threshold) in zip(RULES, scores):
            alarm, _, _ = confirmed_alarm(record["clock"], score, threshold)
            outcome = score_alarm(record, alarm, annotation, rule, row.role)
            rows.append(dict(outcome, dataset=row.dataset, batch=row.batch))
    return pd.DataFrame(rows)


def evaluation_timing(evaluation, histories):
    """Normalize warning leads and quantify certified-stage coverage after alarm."""
    spans = {(batch, sid): h["record"]["tau"] - h["record"]["clock"][0]
             for batch, items in histories.items() for sid, h in items.items()}
    evaluation = evaluation.copy()
    evaluation["normalized_lead"] = [
        row.lead / spans[(row.batch, _raw_id(row.batch, row.id))]
        for row in evaluation.itertuples(index=False)
    ]
    timing = evaluation[evaluation.method.eq("G4")].copy()
    coverage_rows = []
    for row in timing.itertuples(index=False):
        initial_coverage, total_coverage = np.nan, np.nan
        if row.science_valid:
            history = _history(histories, row)
            clock = history["record"]["clock"]
            followup = clock[(clock >= row.alarm_s) & (clock <= row.transition_s)]
            stage = np.zeros(len(followup), dtype=bool)
            for episode in history["annotation"]["intervals"]:
                start, end = episode["start_s"], episode["end_s"]
                member = (followup >= start) & (followup <= end)
                if start <= row.alarm_s <= end:
                    initial_coverage = member.mean()
                stage |= member
            total_coverage = stage.mean()
        coverage_rows.append((initial_coverage, total_coverage))
    timing[["first_episode_post_alarm_coverage", "post_alarm_episode_coverage"]] = coverage_rows
    return evaluation, timing


def matched_variance(batteries, histories):
    """Compare capacity and state variance ratios over the same complete windows."""
    rows = []
    for row in batteries.itertuples(index=False):
        rec = _history(histories, row)["record"]
        clock, knee = rec["clock"], rec["tau"]
        window = CONFIGS[row.batch].variance_window
        values = np.stack([rec["output"], rec["state"]])
        windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=-1)
        complete = np.isfinite(windows).all(axis=(0, 2))
        windows_clock = clock[window - 1:]
        result = dict(batch=row.batch, dataset=row.dataset, id=rec["id"],
                      variance_window=window, tau=knee)
        for period, (lo, hi) in comparison_intervals(clock, knee).items():
            available = complete & (windows_clock >= lo) & (windows_clock < hi)
            variances = np.var(windows[:, available, :], axis=2, ddof=1)
            result[period + "_count"] = int(available.sum())
            for signal, variance in zip(("output", "state"), variances):
                result[f"{signal}_{period}"] = np.median(variance) if len(variance) else np.nan
        valid = (clock[0] < knee < clock[-1] and min(result["early_count"], result["near_count"]) >= 3
                 and min(result["output_early"], result["state_early"]) > 0)
        result["evaluable"] = valid
        for signal in ("output", "state"):
            result[signal + "_amplification"] = (
                result[signal + "_near"] / result[signal + "_early"] if valid else np.nan)
        output_ratio = result["output_amplification"]
        result["relative_amplification"] = (
            result["state_amplification"] / output_ratio if output_ratio > 0 else np.nan)
        rows.append(result)
    return pd.DataFrame(rows)


def rates(outcomes, groups):
    """Compute stage effectiveness among timely alarms for each group."""
    summary = outcomes.groupby(list(groups), sort=False).agg(
        n=("id", "size"), warnings=("warning", "sum"), science_valid=("science_valid", "sum"))
    summary["warning_rate"] = summary.warnings / summary.n
    summary["science_effectiveness"] = summary.science_valid / summary.warnings
    return summary.reset_index()


def comparison_matrix(evaluation):
    """Preserve the display order and calculate outcome colours from each alarm."""
    frame = evaluation[evaluation.dataset.isin(DATASETS) & evaluation.method.isin(("G4", *METHODS[1:]))]
    codes = np.select(
        [frame.at_transition, frame.science_valid, frame.unknown_warning, frame.warning, frame.late],
        [3, 0, 2, 1, 4], default=5,
    )
    order = pd.read_csv(MATRIX_ORDER, dtype={"id": str})
    keys = ["dataset", "id"]
    ordered = pd.MultiIndex.from_frame(order[keys])
    data = frame.assign(code=codes, method=frame.method.replace({"G4": "Proposed"}))
    matrix = data.pivot(index="method", columns=keys, values="code").reindex(index=METHODS, columns=ordered)
    if matrix.isna().any().any() or len(order) != len(data) / len(METHODS):
        raise ValueError("The outcome matrix must contain one evaluation result per cell and method.")
    matrix.columns = [f"{dataset}__{sid}" for dataset, sid in ordered]
    return matrix, order.dataset.to_numpy()


def fatigue_intervals(record):
    start, end = record["transition_fit_clock"][0], record["tau"]
    span = end - start
    return {"earlier": (start + .2 * span, start + .4 * span),
            "near": (start + .8 * span, end)}


def _fatigue_precursor(record, window):
    clock = record["clock"]
    variance = raw_variance(record["state"], window)
    intervals = fatigue_intervals(record)
    summaries = {}
    for period, (lo, hi) in intervals.items():
        selected = variance[(clock >= lo) & (clock < hi)]
        valid = selected[np.isfinite(selected)]
        summaries[period] = (float(np.median(valid)) if len(valid) else np.nan, len(valid))
    return variance, summaries


def _linear_recovery(clock, state, window, transition):
    """Estimate the fatigue example's AR(1) term jointly with a linear time trend."""
    phi = np.full(len(clock), np.nan)
    for end in np.flatnonzero((np.arange(len(clock)) >= window - 1) & (clock <= transition)):
        sl = slice(end - window + 1, end + 1)
        y = state[sl]
        t = clock[sl]
        if not np.isfinite(y).all() or y.std() == 0:
            continue
        y = (y - y.mean()) / y.std()
        t = (t - t.mean()) / (t[-1] - t[0])
        design = np.column_stack([np.ones(window - 1), t[1:], y[:-1]])
        if np.linalg.matrix_rank(design) == design.shape[1]:
            phi[end] = np.linalg.lstsq(design, y[1:], rcond=None)[0][-1]
    kappa = np.full(len(clock), np.nan)
    stable = (phi > 0) & (phi < 1)
    kappa[stable] = -np.log(phi[stable]) / np.diff(clock, prepend=np.nan)[stable]
    return pd.DataFrame(dict(clock=clock, phi=phi, kappa=kappa))


def fatigue_diagnostics(outcomes, histories):
    """Return the specimen examples and calibration-cohort variance contrasts."""
    records = histories[FATIGUE]
    precursor = records["129"]["record"]
    variance, _ = _fatigue_precursor(precursor, 10)
    earlier, near = fatigue_intervals(precursor).values()
    iron = dict(clock=precursor["clock"], variance=variance, transition=precursor["tau"],
                early_start=earlier[0], early_end=earlier[1], late_start=near[0], late_end=near[1])

    rows = []
    selected = outcomes[(outcomes.dataset == FATIGUE) & (outcomes.method == "G4")
                        & (outcomes.role == "calibration_40pct")]
    for sid in selected.id:
        record = records[sid]["record"]
        for window in (10, 30, 60):
            _, summary = _fatigue_precursor(record, window)
            early_var, early_finite = summary["earlier"]
            near_var, near_finite = summary["near"]
            rows.append(dict(id=sid, window=window, early_var=early_var, near_var=near_var,
                             variance_ratio=near_var / early_var, increased=near_var > early_var,
                             early_finite=early_finite, late_finite=near_finite))
    ratios = pd.DataFrame(rows)
    if ratios.groupby("window").increased.agg(["size", "sum"]).values.tolist() != [[44, 38], [44, 37], [44, 36]]:
        raise ValueError("Fatigue variance counts differ from the manuscript.")
    record = records["119"]["record"]
    clock, state = record["clock"], record["state"]
    intervals = fatigue_intervals(record)
    recovery = _linear_recovery(clock, state, 120, record["tau"])
    kappas, means = [], []
    for period, (lo, hi) in intervals.items():
        idx = np.flatnonzero((clock >= lo) & (clock < hi))
        kappas.append(dict(period=period, kappa_median=float(recovery.kappa.iloc[idx].median())))
        raw, trend, residual = variance_components(clock, state, 10, idx).mean(axis=0)
        means.append(dict(period=period, raw=raw, trend=trend, residual=residual))
    for row in means:
        row["residual_relative"] = row["residual"] / means[0]["raw"]
    recovery_example = dict(intervals=intervals, transition=record["tau"],
                            recovery_primary_statistics=kappas, mean_variances=means)
    return iron, ratios, recovery, recovery_example
