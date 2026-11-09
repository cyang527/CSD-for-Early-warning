"""Compute stage labels and warnings from the prepared observations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ews_core as core
from battery_models import identify_battery_parameters, two_segment_linear_transition
from frozen_settings import CITYU_SETTINGS
from softiron_models import load_softiron_records
from specimen_split import fixed_split

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "submission_inputs" / "replay"
FATIGUE_INPUT = ROOT.parent / "submission_inputs" / "softiron_prepared"
FATIGUE = "Soft_Magnetic_Iron"
BATCHES = ["MIT", "HUST1", "HUST2", "CityU", "ISEA", "Tongji", FATIGUE]


def settings(batch):
    if batch == "CityU":
        return (CITYU_SETTINGS["config"], CITYU_SETTINGS["stage_scales"],
                CITYU_SETTINGS["thresholds"])
    frozen, scales = core.read_frozen_configuration(batch)
    path = ROOT / "ablation" / "frozen_results" / f"thresholds_{batch}.json"
    return frozen["config"], scales, json.loads(path.read_text())


def records(batch):
    if batch == FATIGUE:
        return load_softiron_records(FATIGUE_INPUT)

    result = []
    if batch == "CityU":
        frame = pd.read_csv(INPUT / "CityU_capacity.csv")
        for sid, cell in frame.groupby("cell"):
            cell = cell.sort_values("raw_cycle")
            clock = cell.cycle_since_REF.to_numpy(float)
            output = cell.SOH_pct.to_numpy() / 100
            knee = two_segment_linear_transition(clock, output)
            result.append(dict(id=sid, clock=clock, output=output,
                               tau=clock[knee], n_engineering=len(output)))
    else:
        manifest = pd.read_csv(INPUT / f"{batch}.csv", dtype={"id": str})
        with np.load(INPUT / f"{batch}.npz") as arrays:
            for cell in manifest.itertuples():
                result.append(dict(id=cell.id, tau=cell.tau, n_engineering=cell.n_engineering,
                                   clock=arrays[cell.id + "_clock"], output=arrays[cell.id + "_capacity"]))
    for cell in result:
        cell["dataset"] = batch
        cell["state"] = identify_battery_parameters(cell["output"], batch)["A"]
    return result


def run(batch):
    specimens = records(batch)
    roles = fixed_split(specimens).set_index("id").role
    config, scales, thresholds = settings(batch)
    rows = []
    histories = {}
    for cell in specimens:
        sid, clock, state = cell["id"], cell["clock"], cell["state"]
        replay = core.stage_replay(state, clock, scales)
        label = core.stage_annotation(clock, cell["tau"], scales, replay)
        output = cell["output"]
        if batch == FATIGUE:
            # Compare representations on the same one-second grid and finite-state mask.
            with np.load(FATIGUE_INPUT / "features" / f"{int(sid):03d}.npz") as archive:
                features = archive["data"]
            output = np.full(len(clock), np.nan)
            indices = (features[:, 0] + 1 - clock[0]).astype(int)
            output[indices] = features[:, 11]
            output[~np.isfinite(state)] = np.nan

        for signal, simple, full in [(output, "G1", "G2"), (state, "G3", "G4")]:
            trace = core.persistence_detector(signal, config)
            simple_score = trace["variance"] / trace["reference"]
            simple_score[~np.isfinite(trace["score"])] = np.nan
            for group, score in [(simple, simple_score), (full, trace["score"])]:
                alarm, _, _ = core.confirmed_alarm(clock, score, thresholds[group])
                rows.append(core.score_alarm(cell, alarm, label, group, roles[sid]))
            if full == "G4":
                histories[sid] = dict(record=cell, annotation=label, score=trace["score"])

        direction = "up" if batch == FATIGUE else "down"
        for method in core.METHODS[1:]:
            hit = core.comparison_detector(cell["output"], method, direction)["alarm_index"]
            alarm = None if hit is None else float(cell.get("output_clock", clock)[hit])
            rows.append(core.score_alarm(cell, alarm, label, method, roles[sid]))
    frame = pd.DataFrame(rows)
    frame["annotation_evaluable"] = frame.id.map(
        lambda sid: histories[sid]["annotation"]["evaluable"])
    return frame, histories
