"""Notebook-configurable dataset loading.

Battery releases are read from DATA_ROOT and A is reidentified. The supplied
submission_inputs/softiron_prepared archive supplies the fatigue features and
C2 coefficients; warning decisions are recomputed. Optional raw-workbook
identification is available through softiron_models and preprocess_softiron.
"""
from pathlib import Path
import os

from battery_models import load_battery_records
from softiron_models import load_softiron_records


DATA_ROOT = Path(os.environ.get(
    'EWS_DATA_ROOT', str(Path(__file__).resolve().parent.parent / 'data'))).expanduser()
BATTERY_ROOT = DATA_ROOT


def prepared_fatigue_root(data_root=DATA_ROOT):
    """Prefer an explicit path, then data_root/Soft_Magnetic_Iron, then supplied inputs."""
    explicit = os.environ.get('EWS_SOFTIRON_ROOT')
    if explicit:
        return Path(explicit).expanduser()
    external = Path(data_root).expanduser() / 'Soft_Magnetic_Iron'
    supplied = Path(__file__).resolve().parent.parent / 'submission_inputs' / 'softiron_prepared'
    return external if external.is_dir() else supplied


AUDIT_ROOT = prepared_fatigue_root(DATA_ROOT)


def load_records(dataset, battery_root=BATTERY_ROOT, audit_root=AUDIT_ROOT):
    """Load battery capacity or prepared fatigue observations."""
    if dataset == 'Soft_Magnetic_Iron':
        root = Path(audit_root)
        required = ('features', 'pretransition_state_screen/c2_cohort',
                    'specimen_split/split_manifest.csv',
                    'transition_reference/transition_comparison.csv')
        absent = [str(root / name) for name in required if not (root / name).exists()]
        if absent:
            raise FileNotFoundError(
                'Set EWS_SOFTIRON_ROOT to a prepared fatigue directory. Missing: '
                + ', '.join(absent))
        return load_softiron_records(root, recompute_transition=True)
    root = Path(battery_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f'Set EWS_DATA_ROOT to the battery data directory: {root}')
    return load_battery_records(dataset, root)
