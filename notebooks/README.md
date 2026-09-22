# Notebooks

This folder is reserved for optional exploratory notebooks.

The reproducible project workflow is implemented in `src/` and run through the scripts in `scripts/`. This separation keeps reusable training logic out of notebook cells and makes the project easier to rerun from a terminal.

Suggested optional notebooks:

```text
01_eda.ipynb
02_baseline_cnn_experiment.ipynb
03_resnet50_transfer_learning.ipynb
04_error_analysis.ipynb
```

Do not place downloaded datasets, trained model weights, generated figures, or sensitive credentials in this folder.
