# A critical-slowing-down interpretation of battery capacity-knee precursors

Analysis code for reconstructing degradation-related states, evaluating capacity-knee warnings and generating the study figures. The analysis also covers soft-magnetic-iron fatigue.

## Code overview

| Notebook | Purpose |
| --- | --- |
| [Dataset notebooks](submission_code/) | Reconstruct the battery coefficient `A` from prepared capacities. For fatigue, assemble `C2` and the displacement series from fitted coefficients and observation blocks. Evaluate warnings for each dataset. |
| [All_Datasets_Results.ipynb](submission_code/All_Datasets_Results.ipynb) | Aggregate specimen-level outcomes, warning metrics and certified stage intervals. |
| [Four_Group_Ablation.ipynb](submission_code/ablation/Four_Group_Ablation.ipynb) | Compare output- and state-based monitoring under a simple threshold and the full confirmation rule. |
| [Publication_Figures.ipynb](submission_code/Publication_Figures.ipynb) | Generate the main and supplementary figures. |

The notebooks use Python 3.12. The analysis environment is specified in [requirements.txt](requirements.txt).

This repository contains the analysis source code. Reproducing the study results requires the separate analysis package containing prepared inputs, study-specific settings and specimen assignments.

## Citation
Paper:
```
@article{Yang2026,
  title={A critical-slowing-down interpretation of battery capacity-knee precursors},
  author={Chaoying Yang, XX...},
  journal={xx},
  year={2026},
  doi={xx},
  url={xx},
}
```
