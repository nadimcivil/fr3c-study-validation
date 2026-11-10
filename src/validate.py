"""
Study-wise validation of ML models for split tensile strength of
fiber-reinforced rubberized recycled aggregate concrete (FR3C).

Re-analysis of the dataset from Pal, Ahmed & Yazdani (2025),
Cleaner Materials, doi:10.1016/j.clema.2025.100xxx (see README for citation).

Question: do ML models still beat the ACI-type formula (ft = 0.56*sqrt(fc))
when the test set contains whole studies the model has never seen?

Usage:
    python src/validate.py --data data/tensile_strength_dataset.csv
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42

COLUMNS = ["agg_size", "RCA", "rubber", "fiber", "fiber_type",
           "WC", "plasticizer", "fly_ash", "fc", "ft"]

# Number of experiments per source study, in the order of Appendix 1 of the
# paper. The CSV rows follow this order (checked against per-study fc/ft
# ranges). The total must be 346 for 33 studies.
STUDY_COUNTS = [12, 15, 24, 7, 10, 3, 5, 15, 4, 12, 6, 8, 6, 10, 24, 21, 7,
                18, 6, 7, 10, 9, 7, 12, 3, 4, 4, 7, 3, 5, 16, 36, 10]

RATIO_CUTOFF = 1.3  # ft/sqrt(fc) above this is physically implausible


def load(path):
    df = pd.read_csv(path)
    df.columns = COLUMNS
    num = ["agg_size", "RCA", "rubber", "fiber", "WC", "fly_ash", "fc"]
    df[num] = KNNImputer(n_neighbors=5).fit_transform(df[num])
    study = np.repeat(np.arange(len(STUDY_COUNTS)), STUDY_COUNTS)
    assert len(study) == len(df), (
        f"Expected {len(study)} rows, found {len(df)}. "
        "Study grouping only works for the original 346-row file.")
    X = pd.get_dummies(df.drop(columns=["ft"]),
                       columns=["fiber_type", "plasticizer"], dtype=int)
    return X, df["ft"], study


def models():
    return {
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=2, weights="distance", p=1)),
        "Random Forest": RandomForestRegressor(n_estimators=150,
                                               random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.075,
                                max_depth=10, subsample=0.95,
                                colsample_bytree=0.5, random_state=SEED),
    }


def residual_models():
    """Small models that learn only a correction to the ACI ratio."""
    return {
        "Ridge (ratio)": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Small XGB (ratio)": XGBRegressor(n_estimators=100, max_depth=2,
                                          learning_rate=0.05,
                                          random_state=SEED),
    }


def evaluate(X, y, groups, scheme):
    """Return {name: (R2, MAE)} for one CV scheme ('random' or 'study')."""
    if scheme == "random":
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        kw = {}
    else:
        cv = GroupKFold(n_splits=5)
        kw = {"groups": groups}

    base = np.sqrt(X["fc"].values)
    ratio = y.values / base
    out = {"ACI (0.56*sqrt(fc))": 0.56 * base}

    pr = np.zeros(len(y))
    for tr, te in cv.split(X, y, **kw):
        pr[te] = ratio[tr].mean()
    out["Constant ratio"] = pr * base

    for name, m in models().items():
        out[name] = cross_val_predict(m, X, y, cv=cv, **kw)
    for name, m in residual_models().items():
        out[name] = cross_val_predict(m, X, ratio, cv=cv, **kw) * base

    return {k: (r2_score(y, p), mean_absolute_error(y, p))
            for k, p in out.items()}


def main(path, outdir):
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)

    X, y, study = load(path)
    ratio = y.values / np.sqrt(X["fc"].values)
    flag = ratio > RATIO_CUTOFF
    print(f"Samples with ft/sqrt(fc) > {RATIO_CUTOFF}: {flag.sum()} "
          f"(studies: {sorted(set(study[flag]))})")

    rows = []
    for dataset, mask in [("all data (346)", np.ones(len(y), bool)),
                          (f"flagged removed ({(~flag).sum()})", ~flag)]:
        Xs = X[mask].reset_index(drop=True)
        ys = y[mask].reset_index(drop=True)
        gs = study[mask]
        for scheme in ["random", "study"]:
            for name, (r2, mae) in evaluate(Xs, ys, gs, scheme).items():
                rows.append({"dataset": dataset, "split": scheme,
                             "model": name, "R2": round(r2, 3),
                             "MAE_MPa": round(mae, 3)})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "results", "validation_results.csv"),
               index=False)
    print(res.to_string(index=False))

    # Figure: MAE under random vs study-wise split (all data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (dataset, sub) in zip(axes, res.groupby("dataset", sort=False)):
        piv = sub.pivot(index="model", columns="split", values="MAE_MPa")
        piv = piv.loc[res["model"].unique()]
        piv = piv[["random", "study"]]
        piv.columns = ["random split", "study-wise split"]
        piv.plot.bar(ax=ax, color=["#4C72B0", "#C44E52"], legend=False)
        ax.set_title(dataset)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.8)
        ax.tick_params(axis="x", rotation=40)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    axes[0].set_ylabel("MAE (MPa), 5-fold CV")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(os.path.join(outdir, "figures", "mae_random_vs_study.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tensile_strength_dataset.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    main(a.data, a.out)
