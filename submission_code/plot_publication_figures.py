"""Generate the article figures and export their source data."""
from pathlib import Path
from shutil import copyfile

from manuscript_replay import ROOT

STUDY_OVERVIEW = ROOT.parent / 'submission_inputs/publication_figures/study_overview'


def build_main_figures(output_dir, outcomes, histories_by_batch):
    from figure23 import build_figures
    from publication_figures_456 import build_figures_456

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        name = f'Figure_1.{suffix}'
        copyfile(STUDY_OVERVIEW / name, output_dir / name)
    build_figures(output_dir, histories_by_batch)
    build_figures_456(output_dir, outcomes, histories_by_batch)
    return [output_dir / f'Figure_{number}.pdf' for number in range(1, 7)]


def build_si_figures(output_dir, outcomes, histories_by_batch):
    from plot_supplementary_figures import plot_supplementary_figures

    return plot_supplementary_figures(output_dir, outcomes, histories_by_batch)
