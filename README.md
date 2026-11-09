# Study-wise validation of ML models for FR³C split tensile strength

A re-analysis of the dataset from *Pal, Ahmed & Yazdani (2025), "Data driven
tensile strength prediction for fiber-reinforced rubberized recycled aggregate
concrete using machine learning", Cleaner Materials*.

The original paper reports test R² ≈ 0.95 using a random train/test split and
concludes that ML outperforms code formulas (ACI 318-19, CEB-FIP). This project
asks a stricter question:

> **Does the ML advantage survive when the test set contains whole studies
> (labs, cements, test setups) that the model has never seen?**

## Why this matters

The 346 samples come from 33 published studies. Samples from the same study
share material properties and test conditions, so they are not independent.
With a random split, neighbours from the same study appear in both training
and test sets, and distance- or tree-based models can exploit this. Practical
use (predicting a new mix in a new lab) corresponds to a study-wise split.

## Method

- Data: 346 samples, 9 inputs, target = split tensile strength (MPa).
- Study IDs are reconstructed from the order and sizes in Appendix 1 of the
  paper. The reconstruction was checked by comparing per-study sample counts
  and fc / ft ranges against the appendix for all 33 studies.
- Two 5-fold cross-validation schemes: **random** (`KFold`) and **study-wise**
  (`GroupKFold` on study ID).
- Models: KNN (scaled), Random Forest, XGBoost, plus two small models that
  learn only a correction to the ratio `ft / sqrt(fc)`.
- Baselines: ACI-type formula `ft = 0.56·sqrt(fc)` and a constant ratio
  learned from training data.
- Metrics: R² and MAE (MPa).
- Sensitivity check: 16 samples (all from one study, study index 31) have
  `ft / sqrt(fc) > 1.3` (ft up to 16 MPa at fc ≈ 70–78 MPa), which is
  physically implausible for concrete of this strength. Every analysis is
  reported **with and without** these samples. I have not verified the
  original source of these values, so they are flagged, not declared wrong.

## Findings

Fill in after running `python src/validate.py` (see
`results/validation_results.csv` and `figures/mae_random_vs_study.png`).

<!-- Paste the key rows of the results table here. Suggested summary:
- Random split: ML models MAE ≈ ...  vs ACI ≈ ...
- Study-wise split, all data: ...
- Study-wise split, flagged samples removed: ...
-->

## Limitations

- Only 33 independent studies, so differences between models are noisy.
  Small gaps (a few hundredths of an MPa) should not be interpreted.
- The flag threshold (1.3) was chosen after seeing the results. Both versions
  of every result are reported for that reason.
- ACI comparison uses the common approximation `0.56·sqrt(fc)`, not the full
  code procedure, and the formula was not designed for fibre/rubber/RCA mixes.
- No hyperparameter tuning was performed; hyperparameters follow the paper.
- Study IDs depend on row order of the original CSV.

## Reproduce

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Download `tensile strength dataset.csv` from the authors' repository
(<https://github.com/avi049/predict-tensile-strength-of-FR3C>), save it as
`data/tensile_strength_dataset.csv`, then:

```bash
python src/validate.py --data data/tensile_strength_dataset.csv
```

The dataset is **not** redistributed here; it belongs to the original authors.
Please check their repository for licence terms before reusing it.

## Citation

Pal, A., Ahmed, S., Yazdani, M. (2025). Data driven tensile strength
prediction for fiber-reinforced rubberized recycled aggregate concrete using
machine learning. *Cleaner Materials*. (Complete the DOI from the paper.)

## Author

Nadim Hasan, University of Dhaka
