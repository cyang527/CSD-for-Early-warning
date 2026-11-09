"""Replay the published specimen order by ID, independent of directory traversal."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from ews_core import fixed_split as canonical_split

def fixed_split(records, seed=65):
    if seed != 65:
        return canonical_split(records, seed)
    datasets = {r['dataset'] for r in records}
    if len(datasets) != 1:
        raise ValueError('Expected exactly one dataset')
    dataset = datasets.pop()
    order = pd.read_csv(Path(__file__).with_name('specimen_order.csv'), dtype={'id': str})
    group = order.loc[order.dataset.eq(dataset)].sort_values('position').reset_index(drop=True)
    ids = [str(r['id']) for r in records]
    if len(ids) != len(set(ids)) or group.id.duplicated().any() or set(ids) != set(group.id):
        raise ValueError(f'{dataset}: unexpected, missing or duplicate specimen ID')
    np.testing.assert_array_equal(group.position, np.arange(len(ids)))
    shuffled = np.random.RandomState(seed).permutation(len(ids))
    roles = np.full(len(ids), 'calibration_40pct', dtype=object)
    roles[shuffled[:math.ceil(.6 * len(ids))]] = 'evaluation_60pct'
    np.testing.assert_array_equal(roles, group.role)
    return group[['id', 'dataset', 'seed', 'role']].copy()
