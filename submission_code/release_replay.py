"""Compute specimen outcomes, certified stages and evaluation summaries."""
import json
from pathlib import Path

import pandas as pd
from manuscript_replay import BATCHES, FATIGUE, INPUT, ROOT, run


RESULTS = ROOT / 'results'


def collect(batches=BATCHES):
    """Return verified outcomes and stage histories grouped by batch."""
    expected = pd.read_csv(INPUT / "expected_outcomes.csv", dtype={"id": str})
    frames = []
    histories = {}
    for batch in batches:
        frame, histories[batch] = run(batch)
        frame["batch"] = batch
        if batch in ("HUST1", "HUST2"):
            frame["dataset"] = "HUST"
            frame["id"] = batch + "__" + frame["id"].astype(str)
        keys = ["batch", "id", "method"]
        result = frame.set_index(keys).sort_index()
        reference = expected[expected.batch.eq(batch)].set_index(keys).sort_index()
        pd.testing.assert_frame_equal(result, reference[result.columns], check_dtype=False,
                                      rtol=1e-12, atol=1e-12)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), histories


def reproduce(batches=BATCHES):
    return collect(batches)[0]


def summarize(frame):
    evaluation = frame[frame.role.eq("evaluation_60pct")]
    summary = evaluation.groupby(["dataset", "method"], sort=False).agg(
        specimens=("id", "size"), by_transition=("warning", "sum"),
        at_transition=("at_transition", "sum"), aligned=("science_valid", "sum"))
    summary["warning_rate"] = summary.by_transition / summary.specimens
    summary["stage_effectiveness"] = summary.aligned / summary.by_transition
    return summary


def write_results(outcomes, histories, directory=RESULTS / 'All_Datasets'):
    """Write the evaluation rows, pooled summary and all certified intervals."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    evaluation = outcomes[outcomes.role.eq('evaluation_60pct')]
    batteries = evaluation[evaluation.dataset.ne(FATIGUE)].assign(dataset='All batteries')
    summary = summarize(pd.concat([evaluation, batteries], ignore_index=True))
    summary['science_yield'] = summary.aligned / summary.specimens
    summary = summary.rename(columns={
        'specimens': 'n', 'by_transition': 'warnings', 'aligned': 'science_valid',
        'stage_effectiveness': 'science_effectiveness',
    })
    stage_rows = []
    for batch, records in histories.items():
        hust = batch in ('HUST1', 'HUST2')
        for sid, history in records.items():
            stage_rows.append(dict(dataset='HUST' if hust else batch, batch=batch,
                                   id=f'{batch}__{sid}' if hust else sid,
                                   **history['annotation']))

    evaluation.to_csv(directory / 'specimen_outcomes.csv', index=False)
    summary.to_csv(directory / 'summary.csv')
    (directory / 'stage_intervals.json').write_text(
        json.dumps(stage_rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return directory
