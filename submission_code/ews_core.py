"""Trailing-variance warning rules and specimen-level evaluation.

The detector uses state histories. Fixed output references and
retrospectively certified stage intervals define the evaluation.
"""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from frozen_settings import PARAMETERS, STAGE_SCALES


CODE_ROOT = Path(__file__).resolve().parent
DATASETS = ('MIT', 'HUST1', 'HUST2', 'ISEA', 'Tongji', 'Soft_Magnetic_Iron')
METHODS = ('Proposed', 'MR', 'PM', 'IQRM', 'TSM')
DISPLAY_NAMES = {'Soft_Magnetic_Iron': 'Soft magnetic iron',
                 'Five_batteries_combined': 'All batteries'}


def read_frozen_configuration(dataset):
    """Return detector settings and variance-stage scales."""
    return PARAMETERS[dataset], STAGE_SCALES[dataset]


def raw_variance(state, window):
    """Trailing sample variance with complete windows and explicit constants."""
    series = pd.Series(np.asarray(state, float))
    value = series.rolling(window, min_periods=window).var(ddof=1).to_numpy().copy()
    constant = series.rolling(window).max().to_numpy() == series.rolling(window).min().to_numpy()
    value[constant] = 0.0
    value[np.isfinite(value) & (value < 0)] = 0.0
    return value


def smooth_variance(state, window, mean_window):
    """Both operations are trailing; no interpolation or detrending is used."""
    value = raw_variance(state, window)
    return pd.Series(value).rolling(mean_window, min_periods=mean_window).mean().to_numpy()


def historical_reference(value, length):
    """Estimate a stable background from preceding observations.

    Each candidate ends at i-1. Its three block medians span at most
    log(2) on the log scale, with at least 90% finite positive values.
    The background updates when a qualifying estimate is sufficiently lower.
    """
    length = max(9, 3 * (int(length) // 3))
    block = length // 3
    positive = np.isfinite(value) & (value > 0)
    log_value = np.where(positive, np.log(np.where(positive, value, 1)), np.nan)
    med = pd.Series(log_value).rolling(block, min_periods=math.ceil(.9 * block)).median()
    blocks = np.array([med.to_numpy(), med.shift(block).to_numpy(), med.shift(2 * block).to_numpy()])
    span = blocks.max(axis=0) - blocks.min(axis=0)
    count = pd.Series(positive.astype(np.int8)).rolling(length).sum().to_numpy()
    level = pd.Series(np.where(positive, value, np.nan)).rolling(
        length, min_periods=math.ceil(.9 * length)).median().to_numpy()
    stable = (span <= math.log(2)) & (count >= .9 * length) & np.isfinite(level) & (level > 0)
    reference = np.full(len(value), np.nan)
    baseline = np.nan
    for index in range(1, len(value)):
        if stable[index - 1] and (not np.isfinite(baseline) or level[index - 1] < baseline / 1.5):
            baseline = float(level[index - 1])
        reference[index] = baseline
    return reference


def persistence_detector(state, config):
    """The frozen battery and fatigue persistence rules in one parameter API.

    Batteries: recent lower decile above an absolute threshold AND adjacent
    non-overlapping median windows meeting a net-growth factor.
    Fatigue: short median continuously high, with a stronger long median,
    relative to the history-only reference. The caller applies the threshold.
    Missing current evidence cannot trigger a fresh decision.
    """
    value = smooth_variance(state, int(config['w']), int(config['m']))
    reference = historical_reference(value, config['b']) if config['b'] else np.ones(len(value))
    duration = int(config['d'])
    series = pd.Series(value)
    median_series = series.rolling(duration, min_periods=math.ceil(.9 * duration)).median()
    median = median_series.to_numpy()
    previous = median_series.shift(duration).to_numpy()
    gain = np.divide(median, previous, out=np.full(len(value), np.nan),
                     where=np.isfinite(previous) & (previous > 0))
    if int(config['h']) == 1:
        persistence = series.rolling(duration, min_periods=math.ceil(.9 * duration)).quantile(.1).to_numpy()
    else:
        short = max(2, math.ceil(duration / 3))
        broad = series.rolling(short, min_periods=math.ceil(.8 * short)).median()
        persistence = broad.rolling(duration, min_periods=duration).min().to_numpy()
    boundary = np.minimum(persistence, median / float(config['r']))
    score = np.divide(boundary, reference, out=np.full(len(value), np.nan),
                      where=np.isfinite(reference) & (reference > 0))
    if config['growth'] > 0:
        score = np.where(gain >= config['growth'], score, 0.0)
        score[~np.isfinite(gain)] = np.nan
    unavailable = (~np.isfinite(value) | ~np.isfinite(persistence)
                   | ~np.isfinite(median) | ~np.isfinite(reference))
    score[unavailable] = np.nan
    return dict(raw_variance=raw_variance(state, int(config['w'])), variance=value,
                reference=reference, gain=gain, score=score)


def confirmed_alarm(clock, score, threshold):
    """Record the first actual confirmation, then latch permanently at one."""
    hit = np.flatnonzero(np.isfinite(score) & (score > threshold))
    index = int(hit[0]) if len(hit) else None
    alarm = None if index is None else float(clock[index])
    binary = np.zeros(len(clock), np.uint8)
    if index is not None:
        binary[index:] = 1
    return alarm, index, binary


def fixed_split(records, seed=65):
    """Assign specimens to the 40% calibration and 60% evaluation groups."""
    dataset = records[0]['dataset']
    ordered = sorted(records, key=(lambda r: int(r['id'])) if dataset == 'Soft_Magnetic_Iron'
                     else (lambda r: str(r['id'])))
    permutation = np.random.RandomState(seed).permutation(len(ordered))
    evaluation = {str(ordered[i]['id']) for i in permutation[:math.ceil(.6 * len(ordered))]}
    return pd.DataFrame([dict(id=str(r['id']), dataset=dataset, seed=seed,
                              role='evaluation_60pct' if str(r['id']) in evaluation else 'calibration_40pct')
                         for r in ordered])


def annotation_map(records, manifest, scales):
    """Compute specimen-level variance-stage annotations shared by the warning methods."""
    roles = manifest.set_index('id')['role'].to_dict()
    entries = {}
    for record in records:
        role, sid = roles[str(record['id'])], str(record['id'])
        replay = stage_replay(record['state'], record['clock'], scales)
        annotation = stage_annotation(record['clock'], record['tau'], scales, replay)
        annotation.update(dataset=record['dataset'], role=role, id=sid)
        entries[(record['dataset'], role, sid)] = annotation
    return entries


def stage_replay(state, clock, scales):
    """Track variance-stage activation and certification on the observation clock.

    The statistic is a W-point trailing mean of W-point sample variance.
    Background updates pause during active episodes.
    """
    window = int(scales['variance_window'])
    series = pd.Series(state)
    variance = series.rolling(window).var(ddof=1).to_numpy()
    value = (np.full(len(state), np.nan) if len(state) < window else
             np.r_[np.full(window - 1, np.nan), np.convolve(variance, np.ones(window) / window, mode='valid')])
    length, median_length, duration = (int(scales[k]) for k in
                                      ('background_points', 'stage_median_points', 'duration_points'))
    block = length // 3
    positive = np.isfinite(value) & (value > 0)
    logs = np.where(positive, np.log(np.where(positive, value, 1)), np.nan)
    med = pd.Series(logs).rolling(block, min_periods=math.ceil(.9 * block)).median()
    blocks = np.array([med.to_numpy(), med.shift(block).to_numpy(), med.shift(2 * block).to_numpy()])
    span = blocks.max(axis=0) - blocks.min(axis=0)
    count = pd.Series(positive.astype(int)).rolling(length).sum().to_numpy()
    level = pd.Series(np.where(positive, value, np.nan)).rolling(
        length, min_periods=math.ceil(.9 * length)).median().to_numpy()
    stable = (span <= np.log(2)) & (count >= .9 * length) & np.isfinite(level) & (level > 0)
    broad = pd.Series(value).rolling(median_length, min_periods=math.ceil(.8 * median_length)).median().to_numpy()
    strong = pd.Series(value).rolling(duration, min_periods=math.ceil(.9 * duration)).median().to_numpy()
    strong_count = pd.Series(value).rolling(duration).count().to_numpy()
    background, ratio = np.full(len(value), np.nan), np.full(len(value), np.nan)
    active, first_certificate = np.zeros(len(value), bool), np.zeros(len(value), bool)
    base, elapsed, episode_certified, updates = np.nan, 0, False, []
    for index in range(1, len(value)):
        if elapsed == 0 and stable[index - 1] and (not np.isfinite(base) or level[index - 1] < base / 1.5):
            base = float(level[index - 1])
            updates.append(dict(index=index, clock=float(clock[index]), baseline=base))
        background[index] = base
        if not np.isfinite(base):
            continue
        ratio[index] = value[index] / base
        active[index] = bool(np.isfinite(broad[index]) and broad[index] >= 1.5 * base)
        if not active[index]:
            elapsed, episode_certified = 0, False
            continue
        elapsed += 1
        qualifies = (elapsed >= duration and strong_count[index] >= math.ceil(.9 * duration)
                     and strong[index] >= 2 * base)
        first_certificate[index] = bool(qualifies and not episode_certified)
        episode_certified = bool(episode_certified or qualifies)
    return dict(variance=value, background=background, ratio=ratio, active=active,
                first_certificate=first_certificate, updates=updates)


def stage_annotation(clock, transition, scales, replay):
    """Build the certified intervals at or before the fixed output reference."""
    before = np.asarray(clock) <= float(transition)
    mask = replay['active'] & before
    changes = np.diff(np.r_[False, mask, False].astype(int))
    intervals = []
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        certificates = np.flatnonzero(replay['first_certificate'][start:end])
        if not len(certificates):
            continue
        at = start + int(certificates[0])
        intervals.append(dict(start_s=float(clock[start]), end_s=float(clock[end - 1]),
                              certified_s=float(clock[at]), background=float(replay['background'][at]),
                              duration_points=int(end - start)))
    updates = [u for u in replay['updates'] if u['clock'] <= transition]
    ready = updates[0]['index'] if updates else None
    value = replay['variance']
    enough = False if ready is None else int((before & (np.arange(len(clock)) >= ready)
                                             & np.isfinite(value) & (value > 0)).sum()) >= scales['duration_points']
    evaluable = bool(intervals or enough)
    reason = 'ok' if evaluable else 'insufficient_followup' if ready is not None else 'no_stable_background_before_transition'
    return dict(evaluable=evaluable, reason=reason, intervals=intervals)


def comparison_detector(output, method, direction, fraction=.30, run=10):
    """Fit an output-residual baseline to the first fraction of the retained record.

    The alarm is confirmed at the last of the required consecutive strict
    crossings. Capacity crosses down and mean displacement crosses up.
    """
    output = np.asarray(output, float)
    length, index = len(output), np.arange(len(output), dtype=float)
    baseline_length = int(np.floor(fraction * length))
    finite = np.isfinite(output[:baseline_length])
    if finite.sum() < 3:
        return dict(alarm_index=None, boundary=np.full(length, np.nan), K=baseline_length,
                    finite_fit_points=int(finite.sum()))
    coefficients = np.polyfit(index[:baseline_length][finite], output[:baseline_length][finite], 1)
    trend = np.polyval(coefficients, index)
    residual = output[:baseline_length][finite] - trend[:baseline_length][finite]
    q1, q3 = np.percentile(residual, [25, 75])
    offsets = (dict(MR=min(residual), PM=np.percentile(residual, 5), IQRM=q1 - 1.5 * (q3 - q1),
                    TSM=np.mean(residual) - 3 * np.std(residual, ddof=1)) if direction == 'down' else
               dict(MR=max(residual), PM=np.percentile(residual, 95), IQRM=q3 + 1.5 * (q3 - q1),
                    TSM=np.mean(residual) + 3 * np.std(residual, ddof=1)))
    boundary = trend + offsets[method]
    beyond = output < boundary if direction == 'down' else output > boundary
    count, alarm_index = 0, None
    for current in range(baseline_length, length):
        count = count + 1 if beyond[current] else 0
        if count == run:
            alarm_index = current
            break
    return dict(alarm_index=alarm_index, boundary=boundary, K=baseline_length,
                finite_fit_points=int(finite.sum()))


def score_alarm(record, alarm, annotation, method, role):
    """Common science criterion and separate engineering timing criteria."""
    transition = float(record['tau'])
    lead = None if alarm is None else transition - alarm
    warning = bool(lead is not None and lead >= 0)
    stage_valid = bool(warning and annotation['evaluable'] and any(
        episode['start_s'] <= alarm <= episode['end_s'] for episode in annotation['intervals']))
    original_valid = bool(warning and lead > 0 and lead <= .2 * record['n_engineering'])
    engineering_valid = original_valid if record['dataset'] != 'Soft_Magnetic_Iron' else bool(warning and lead >= 50)
    return dict(dataset=record['dataset'], id=str(record['id']), role=role, method=method,
                alarm_s=alarm, transition_s=transition, lead=lead, warning=warning,
                coordinate_unit='s' if record['dataset'] == 'Soft_Magnetic_Iron' else 'retained battery observations',
                science_valid=stage_valid, engineering_valid=engineering_valid,
                original_20pct_valid=original_valid, late=bool(lead is not None and lead < 0),
                at_transition=bool(lead == 0), missed=alarm is None,
                unknown_warning=bool(warning and not annotation['evaluable']))


def summarize(outcomes):
    """Retain all specimens and unknown annotations in the declared denominators."""
    records = outcomes.to_dict('records') if isinstance(outcomes, pd.DataFrame) else outcomes
    length = len(records)
    count = lambda key: sum(bool(r[key]) for r in records)
    warnings, stages = count('warning'), count('science_valid')
    engineering, original = count('engineering_valid'), count('original_20pct_valid')
    return dict(n=length, warnings=warnings, warning_rate=warnings / length if length else np.nan,
                science_valid=stages, science_effectiveness=stages / warnings if warnings else 0.,
                science_yield=stages / length if length else np.nan,
                engineering_valid=engineering, engineering_precision=engineering / warnings if warnings else 0.,
                engineering_yield=engineering / length if length else np.nan,
                original_20pct_valid=original, original_20pct_precision=original / warnings if warnings else 0.,
                unknown_warnings=count('unknown_warning'), late=count('late'), missed=count('missed'),
                at_transition=count('at_transition'))


def evaluate_records(records, frozen, manifest, annotations):
    """Apply the proposed rule and the four output-residual comparators."""
    roles = manifest.set_index('id')['role'].to_dict()
    outcomes, traces, baselines = [], {}, []
    for record in records:
        sid, dataset = str(record['id']), record['dataset']
        role = roles[sid]
        annotation = annotations[(dataset, role, sid)]
        trace = persistence_detector(record['state'], frozen['config'])
        alarm, alarm_index, binary = confirmed_alarm(record['clock'], trace['score'], frozen['threshold'])
        trace.update(alarm_s=alarm, alarm_index=alarm_index, binary=binary)
        traces[sid] = trace
        outcomes.append(score_alarm(record, alarm, annotation, 'Proposed', role))
        output, clock = record['output'], record.get('output_clock', record['clock'])
        for method in METHODS[1:]:
            result = comparison_detector(output, method, 'up' if dataset == 'Soft_Magnetic_Iron' else 'down')
            index = result['alarm_index']
            at = None if index is None else float(clock[index])
            outcomes.append(score_alarm(record, at, annotation, method, role))
            baselines.append(dict(dataset=dataset, id=sid, role=role, method=method, K=result['K'],
                                  n_output=len(output), finite_fit_points=result['finite_fit_points'],
                                  baseline_fraction=.30, confirmation_points=10))
    frame = pd.DataFrame(outcomes)
    summary = pd.DataFrame([dict(dataset=dataset, role=role, method=method, **summarize(group))
                            for (dataset, role, method), group in frame.groupby(
                                ['dataset', 'role', 'method'], sort=False)])
    return frame, summary, traces, pd.DataFrame(baselines)


def display_metrics(summary):
    """Main manuscript metrics, with engineering precision shown separately."""
    columns = ['dataset', 'method', 'n', 'warnings', 'warning_rate', 'science_valid',
               'science_effectiveness', 'engineering_precision', 'original_20pct_precision',
               'late', 'at_transition', 'missed', 'unknown_warnings']
    shown = summary[columns].rename(columns={'science_effectiveness': 'Stage effectiveness',
                                            'engineering_precision': 'Engineering precision',
                                            'original_20pct_precision': '20% timing precision',
                                            'warning_rate': 'Warning rate'}).copy()
    for column in ('Warning rate', 'Stage effectiveness', 'Engineering precision', '20% timing precision'):
        shown[column] = shown[column].map(lambda value: f'{value:.1%}')
    shown['dataset'] = shown['dataset'].replace(DISPLAY_NAMES)
    return shown


def check_result_consistency(outcomes):
    """Check lead calculations and evaluate the outcome flags."""
    actual = outcomes.loc[outcomes.role.eq('evaluation_60pct')].copy()
    assert not actual.duplicated(['dataset', 'id', 'method']).any()
    assert (actual.groupby(['dataset', 'id']).size() == 5).all()
    lead = actual.transition_s - actual.alarm_s
    np.testing.assert_allclose(actual.lead, lead, rtol=0, atol=0, equal_nan=True)
    assert np.array_equal(actual.warning, (lead >= 0).fillna(False))
    assert np.array_equal(actual.missed, actual.alarm_s.isna())
    for flag in ('science_valid', 'engineering_valid', 'original_20pct_valid'):
        assert not (actual[flag] & ~actual.warning).any(), flag
    return dict(evaluation_method_records=len(actual), metric_flags_consistent=True,
                timing_flags_verified=True)


def prefix_checks(records, config, traces):
    """Check history-only downstream calculations at multiple real prefixes."""
    checked = 0
    for record in records:
        full = traces[str(record['id'])]
        stops = {min(100, len(record['state']) - 1), min(300, len(record['state']) - 1)}
        if full['alarm_index'] is not None:
            stops.update([full['alarm_index'], full['alarm_index'] + 1])
        for stop in sorted(stops):
            if stop <= 1 or stop >= len(record['state']):
                continue
            cut = persistence_detector(record['state'][:stop], config)
            np.testing.assert_allclose(cut['score'], full['score'][:stop], rtol=1e-7, atol=1e-15, equal_nan=True)
            checked += 1
        index = full['alarm_index']
        if index is not None:
            assert not full['binary'][:index].any() and full['binary'][index:].all()
    return dict(real_prefix_checks=checked, downstream_history_only=True,
                actual_confirmation_and_permanent_latch=True)


def plot_output_collection(records, dataset):
    """Show every output on its original, unnormalized progress coordinate."""
    fig, axis = plt.subplots(figsize=(10, 4.2), layout='constrained')
    for record in records:
        clock = record.get('output_clock', record['clock'])
        axis.plot(clock, record['output'], linewidth=.45, alpha=.22, color='#ae6143' if dataset == 'Soft_Magnetic_Iron' else '#257d91')
    axis.set(xlabel='Recorded time (s)' if dataset == 'Soft_Magnetic_Iron' else 'Retained cycle coordinate',
             ylabel='Mean displacement (mm)' if dataset == 'Soft_Magnetic_Iron' else 'Processed capacity',
             title=f'{DISPLAY_NAMES.get(dataset, dataset)}: all {len(records)} measured outputs')
    axis.spines[['top', 'right']].set_visible(False)
    return fig


def plot_warning_examples(records, manifest, traces, annotations, number=2):
    """Plot the first evaluation specimens in specimen-ID order."""
    selected = set(manifest.loc[manifest.role.eq('evaluation_60pct'), 'id'])
    examples = [r for r in records if str(r['id']) in selected][:number]
    fig, axes = plt.subplots(4, len(examples), figsize=(12, 9), sharex='col', squeeze=False,
                             layout='constrained')
    for column, record in enumerate(examples):
        dataset, sid = record['dataset'], str(record['id'])
        clock, trace = record['clock'], traces[sid]
        color = '#ae6143' if dataset == 'Soft_Magnetic_Iron' else '#257d91'
        out_clock = record.get('output_clock', clock)
        axes[0, column].plot(out_clock, record['output'], color='#607784', lw=.75)
        axes[0, column].set_title(f'{DISPLAY_NAMES.get(dataset, dataset)} | {sid}')
        axes[0, column].set_ylabel('Mean displacement (mm)' if dataset == 'Soft_Magnetic_Iron' else 'Capacity')
        axes[1, column].plot(clock, record['state'], color=color, lw=.7)
        axes[1, column].set_ylabel(r'$C_2$ (mm kN$^{-2}$)' if dataset == 'Soft_Magnetic_Iron' else r'$A$')
        axes[2, column].plot(clock, trace['variance'], color=color, lw=.8)
        axes[2, column].set_ylabel('Trailing state variance')
        axes[2, column].set_yscale('log')
        axes[3, column].step(clock, trace['binary'], where='post', color=color, lw=1.3)
        axes[3, column].set(yticks=[0, 1], ylim=(-.1, 1.1), ylabel='Confirmed alarm',
                            xlabel='Recorded time (s)' if dataset == 'Soft_Magnetic_Iron' else 'Retained cycle coordinate')
        for axis in axes[:, column]:
            axis.axvline(record['tau'], color='#222222', ls='--', lw=.9)
            if trace['alarm_s'] is not None:
                axis.axvline(trace['alarm_s'], color=color, lw=.8)
            for episode in annotations[(dataset, 'evaluation_60pct', sid)]['intervals']:
                axis.axvspan(episode['start_s'], episode['end_s'], color='#86ad92', alpha=.15)
            axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Dashed: fixed output transition | solid: actual alarm | green: frozen variance stage', fontsize=10)
    return fig
