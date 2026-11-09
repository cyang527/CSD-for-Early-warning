"""Compute one-second fatigue features and compliance fits from raw workbooks.

Full cohort:
    python preprocess_softiron.py --raw-root RAW_FOLDER --output-root PREPARED_FOLDER
Single-workbook fit through 120 seconds:
    python preprocess_softiron.py --raw-root RAW_FOLDER --output-root CHECK_FOLDER --ids 119 --end-s 120
"""

from pathlib import Path
import argparse
import json
import math
import re
import zipfile

import numpy as np
import pandas as pd

from softiron_models import (
    COEFFICIENT_NAMES, FEATURE_COLUMNS, _numeric_blocks, _block_moments,
    _identify_moments, historical_output_series, two_segment_linear_transition,
)


MATERIAL = "Soft Magnetic Iron"
# Specimen identifiers for the soft magnetic iron cohort.
# The raw release contains additional soft-iron files outside this cohort.
COHORT_IDS = (
    1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 101, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193,
)
_WORKBOOK = re.compile(r"^(\d+)#\s*Soft Magnetic Iron\s*,", re.IGNORECASE)


def _summarize_features(raw, starts, seconds):
    """Exact extrema, centred moments and 20-Hz projections in each time bin."""
    lengths = np.diff(np.r_[starts, len(raw)])
    ends = starts + lengths - 1
    channels = raw[:, 1:]
    mean = np.add.reduceat(channels, starts, axis=0) / lengths[:, None]
    # Centring before squaring avoids cancellation from displacement offsets.
    centred = channels - np.repeat(mean, lengths, axis=0)
    variance = np.add.reduceat(centred**2, starts, axis=0) / lengths[:, None]
    covariance = np.add.reduceat(centred[:, 0] * centred[:, 1], starts) / lengths
    phase = 2 * np.pi * 20 * np.remainder(raw[:, 0], 1)
    cosine = 2 * np.add.reduceat(centred * np.cos(phase)[:, None], starts, axis=0) / lengths[:, None]
    sine = 2 * np.add.reduceat(centred * np.sin(phase)[:, None], starts, axis=0) / lengths[:, None]
    low = np.minimum.reduceat(channels, starts, axis=0)
    high = np.maximum.reduceat(channels, starts, axis=0)
    return np.column_stack([
        seconds, raw[starts, 0], raw[ends, 0], lengths,
        low[:, 0], high[:, 0], low[:, 1], high[:, 1], low[:, 2], high[:, 2],
        mean, variance, covariance,
        cosine[:, 0], sine[:, 0], cosine[:, 1], sine[:, 1], cosine[:, 2], sine[:, 2],
    ])


def _prepare_workbook(path, specimen_id, end_s=None):
    """Stream one workbook; retain only a final one-second raw carry buffer."""
    source_stat = path.stat()
    if source_stat.st_size == 0:
        raise ValueError(f"Empty raw workbook: {path}")
    features, moments, sheets = [], [], []
    carry = np.empty((0, 4), float)
    previous, raw_rows, gaps = -np.inf, 0, 0
    cutoff_reached = False

    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (name for name in archive.namelist()
             if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
            key=lambda name: int(re.search(r"sheet(\d+)", name)[1]),
        )
        if not members:
            raise ValueError(f"No numerical worksheets: {path}")
        for member in members:
            sheet_rows = 0
            for block in _numeric_blocks(archive, member, specimen_id):
                if block[0, 0] <= previous or np.any(np.diff(block[:, 0]) <= 0):
                    raise ValueError(f"Non-increasing raw time: #{specimen_id}, {member}")
                if end_s is not None:
                    cutoff_reached = bool(block[-1, 0] >= end_s)
                    block = block[block[:, 0] < end_s]
                if not len(block):
                    if cutoff_reached:
                        break
                    continue
                gaps += int(np.sum(np.diff(block[:, 0]) > .02))
                gaps += int(np.isfinite(previous) and block[0, 0] - previous > .02)
                previous = block[-1, 0]
                raw_rows += len(block)
                sheet_rows += len(block)
                joined = np.vstack((carry, block))
                seconds = np.floor(joined[:, 0]).astype(np.int64)
                cuts = np.r_[0, np.flatnonzero(np.diff(seconds)) + 1, len(joined)]
                last_start = cuts[-2]
                carry = joined[last_start:].copy()
                complete = joined[:last_start]
                if len(complete):
                    starts, ends = cuts[:-2], cuts[1:-1]
                    features.append(_summarize_features(complete, starts, seconds[starts]))
                    moment = _block_moments(complete, starts, ends)
                    if len(moment):
                        moments.append(moment)
                if cutoff_reached:
                    break
            sheets.append({"name": member, "rows": sheet_rows})
            if cutoff_reached:
                break

    # Full-workbook preparation retains the final partial bin.
    # With --end-s, only bins completed by that explicit arrival time are kept.
    if len(carry) and (end_s is None or np.floor(carry[0, 0]) + 1 <= end_s):
        starts, ends = np.array([0]), np.array([len(carry)])
        features.append(_summarize_features(carry, starts, np.array([int(np.floor(carry[0, 0]))])))
        moment = _block_moments(carry, starts, ends)
        if len(moment):
            moments.append(moment)
    if not features:
        raise ValueError(f"No usable numerical bins in #{specimen_id}")
    data = np.vstack(features)
    pooled = np.vstack(moments) if moments else np.empty((0, 23), float)
    clock, coefficients = _identify_moments(pooled, identification_seconds=1)
    if len(coefficients) == 0:
        raise ValueError(f"No >=120-point response blocks in #{specimen_id}")
    final_stat = path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError(f"Raw workbook changed during reading: {path}")

    metadata = {
        "id": specimen_id, "file": path.name, "material": MATERIAL,
        "bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns,
        "status": "ok", "raw_rows": raw_rows, "one_second_bins": len(data),
        "sheets": sheets, "gaps_over_20ms": gaps,
        "first_t": float(data[0, 1]), "last_t": float(data[-1, 2]),
        "identification_seconds": 1, "partial_rebuild": end_s is not None,
        "arrival_time_cutoff_s": end_s,
        "final_bin_policy": "retain final partial bin" if end_s is None else "completed by explicit cutoff only",
        "source_workbook": path.name,
    }
    if specimen_id == 2:
        metadata["schema_note"] = "Measured channels are read from columns A:B:C:D."
    elif specimen_id == 3:
        metadata["schema_note"] = "Sheet1 A:B:C:D; later sheets A:C:D:F. F is the source-provided cached magnetic calibration."
    return data, clock, coefficients, metadata


def _reference_row(specimen_id, metadata, features):
    """Recompute only the retrospective output reference and its loading span."""
    prepared = historical_output_series(features)
    fit = two_segment_linear_transition(prepared["transition_clock"], prepared["transition_output"])
    if fit["status"] != "candidate":
        raise ValueError(f"No output reference fit for #{specimen_id}")
    # Engineering scoring uses the loading boundaries before trimming.
    row = {"id": specimen_id, "material": MATERIAL, "file": metadata["file"],
           "new_tau_s": fit["tau"], "loading_start_s": prepared["output_clock"][0] - 5,
           "loading_end_s": prepared["output_clock"][-1] + 5,
           "force_amplitude_kN": prepared["archived_force_reference"],
           "loading_bins": len(prepared["transition_clock"]), **fit}
    return row


def _write_full_cohort_tables(output_root, references):
    """Write seed-65 split and references only after all 110 full records pass."""
    ordered = np.array(COHORT_IDS, int)
    permutation = np.random.RandomState(65).permutation(len(ordered))
    evaluation = set(ordered[permutation[:math.ceil(.6 * len(ordered))]])
    manifest = pd.DataFrame([
        {"id": int(sid), "material": MATERIAL,
         "role": "evaluation_60pct" if sid in evaluation else "calibration_40pct", "seed": 65}
        for sid in ordered
    ])
    labels = pd.DataFrame(references).sort_values("id")
    manifest_path = output_root / "specimen_split" / "split_manifest.csv"
    label_path = output_root / "transition_reference" / "transition_comparison.csv"
    if label_path.exists():
        existing = pd.read_csv(label_path).set_index("id")
        if set(COHORT_IDS).issubset(existing.index):
            for field in ("new_tau_s", "loading_start_s", "loading_end_s"):
                np.testing.assert_array_equal(labels[field].to_numpy(), existing.loc[labels.id, field].to_numpy(),
                                              err_msg=f"Existing fixed reference changed: {field}")
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        existing = existing.loc[existing.material.eq(MATERIAL)].sort_values("id")
        if len(existing) == len(COHORT_IDS):
            assert np.array_equal(existing.id.to_numpy(), manifest.id.to_numpy()), "Existing cohort IDs changed"
            assert np.array_equal(existing.role.to_numpy(), manifest.role.to_numpy()), "Existing split changed"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    labels.to_csv(label_path, index=False)


def preprocess_softiron(raw_root, output_root, ids=None, end_s=None):
    """Prepare the full cohort or a specified subset of workbooks.

    The full run exports the specimen and transition reference tables.
    Use a separate output directory for a run with a time cutoff.
    """
    raw_root, output_root = Path(raw_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw workbook folder not found: {raw_root}")
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("Prepared outputs must be outside the raw workbook folder")
    if end_s is not None and (not np.isfinite(end_s) or end_s <= 0):
        raise ValueError("end_s must be a positive finite arrival-time cutoff")
    selected = set(COHORT_IDS if ids is None else (int(sid) for sid in ids))
    if not selected or not selected.issubset(COHORT_IDS):
        raise ValueError("ids must be a nonempty subset of the fixed 110-specimen manuscript cohort")
    sources = {}
    for path in raw_root.glob("*.xlsx"):
        match = _WORKBOOK.match(path.name)
        if match and int(match[1]) in selected:
            sid = int(match[1])
            if sid in sources:
                raise ValueError(f"Duplicate raw workbooks for #{sid}: {sources[sid]}, {path}")
            sources[sid] = path
    missing = sorted(selected - sources.keys())
    if missing:
        raise FileNotFoundError(f"Missing raw soft-iron workbooks for retained IDs: {missing}")

    feature_root = output_root / "features"
    coefficient_root = output_root / "pretransition_state_screen" / "c2_cohort"
    feature_root.mkdir(parents=True, exist_ok=True)
    coefficient_root.mkdir(parents=True, exist_ok=True)
    summaries, references = [], []
    full_cohort = selected == set(COHORT_IDS) and end_s is None
    for count, sid in enumerate(sorted(selected), start=1):
        data, clock, coefficients, metadata = _prepare_workbook(sources[sid], sid, end_s=end_s)
        feature_path = feature_root / f"{sid:03d}.npz"
        coefficient_path = coefficient_root / f"coefficients_{sid:03d}.npz"
        if end_s is not None and (feature_path.exists() or coefficient_path.exists()):
            raise FileExistsError("Use a new output directory for a time-limited specimen fit.")
        # Ordinary numeric archives, readable with allow_pickle=False.
        np.savez_compressed(feature_path, data=data, columns=np.array(FEATURE_COLUMNS))
        feature_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        np.savez_compressed(coefficient_path, t=clock, coefficients=coefficients[:, None, :],
                            identification_seconds=np.array([1]), columns=np.array(COEFFICIENT_NAMES))
        if full_cohort:
            references.append(_reference_row(sid, metadata, data))
        summary = {"id": sid, "material": MATERIAL, "status": "ok", "raw_rows": metadata["raw_rows"],
                   "one_second_bins": len(data), "coefficient_seconds": len(clock),
                   "finite_C2_seconds": int(np.isfinite(coefficients[:, 2]).sum()),
                   "partial_rebuild": end_s is not None,
                   "features": str(feature_path), "coefficients": str(coefficient_path)}
        summaries.append(summary)
        print(f"Prepared {count}/{len(selected)} | #{sid:03d} | {len(data)} one-second bins", flush=True)
    if full_cohort:
        _write_full_cohort_tables(output_root, references)
    else:
        print("Subset/limited-time rebuild: existing cohort split and output references left unchanged.", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--end-s", type=float, default=None)
    args = parser.parse_args()
    summaries = preprocess_softiron(args.raw_root, args.output_root, args.ids, args.end_s)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
