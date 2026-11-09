"""Run the analysis notebooks with this Python interpreter."""
from pathlib import Path
import argparse
import os
import sys
from tempfile import TemporaryDirectory
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parent


def execute(path):
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    started = time.monotonic()
    client = NotebookClient(notebook, km=manager, timeout=1200, record_timing=False,
                            resources={'metadata': {'path': str(ROOT.parent)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f'{path.relative_to(ROOT)}: completed ({time.monotonic() - started:.1f} s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*')
    args = parser.parse_args()
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else (
        sorted(ROOT.glob('*_EWS.ipynb')) + [ROOT / 'All_Datasets_Results.ipynb',
                                            ROOT / 'ablation/Four_Group_Ablation.ipynb',
                                            ROOT / 'Publication_Figures.ipynb'])
    with TemporaryDirectory(prefix='publication-kernels-') as cache:
        for key in ('MPLCONFIGDIR', 'IPYTHONDIR', 'JUPYTER_RUNTIME_DIR'):
            os.environ[key] = str(Path(cache) / key.lower())
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        for path in paths:
            execute(path)


if __name__ == '__main__':
    main()
