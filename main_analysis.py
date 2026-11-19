"""Main analysis program for the RAMT-CPN comfort prediction model.

Reads input_release.xlsx from this folder, runs the subject-independent
five-fold grouped cross-validation with the fixed 300-round configuration,
and writes the held-out prediction table, the per-fold table, the baseline
comparison and the SHAP attribution to the results folder.

Environment: Python 3.13.5, scikit-learn 1.7.1, XGBoost 3.0.4, SHAP 0.48.0,
pandas 2.3.1, NumPy 2.2.6.
"""

import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "input_release.xlsx")
OUT = os.path.join(HERE, "results")
K = 5
N_ROUNDS = 300
PCA_VAR = 0.85
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.01,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "seed": 42,
}

ANTHRO = ["age", "height", "weight"]
IMPACT = "The instantaneous impact"
L4L5 = "The pressure at lumbar L4/L5"
DL4L5 = "the different of The pressure at lumbar L4/L5 and no load"
T_MF_B = "The MF of the trapezius before use"
T_RVC_B = "The %RVC of the trapezius before use"
T_MF_A = "The MF of the trapezius after use"
T_RVC_A = "The %RVC of the trapezius after use"
T_MF_D = "the different of the trapezius MF between use and after"
T_RVC_D = "the different of the trapezius %RVC between use and after"
E_MF_B = "The MF of the erector spinae before use"
E_RVC_B = "The %RVC of the erector spinae before use"
E_MF_A = "The MF of the erector spinae after use"
E_RVC_A = "The %RVC of the erector spinae after use"
E_MF_D = "the different of the erector spinae MF between use and after"
E_RVC_D = "the different of the erector spinae %RVC between use and after"
M_MF_B = "The MF of the multifidus before use"
M_RVC_B = "The %RVC of the multifidus before use"
M_MF_A = "The MF of the multifidus after use"
M_RVC_A = "The %RVC of the multifidus after use"
M_MF_D = "the different of the multifidus MF between use and after"
M_RVC_D = "the different of the multifidus %RVC between use and after"

REGION = {
    "Shoulder": {
        "sheet": "shoulder",
        "feats": ANTHRO + [IMPACT, T_MF_B, T_RVC_B, T_MF_A, T_RVC_A, T_MF_D, T_RVC_D],
        "targets": ["Shoulder Comfort", "Shoulder Compression", "Shoulder Sore Swelling"],
    },
    "Back": {
        "sheet": "back",
        "feats": ANTHRO + [E_MF_B, E_RVC_B, E_MF_A, E_RVC_A, E_MF_D, E_RVC_D],
        "targets": ["Back Comfort", "Back Compression", "Back Sore Swelling"],
    },
    "Waist": {
        "sheet": "waist",
        "feats": ANTHRO + [L4L5, DL4L5, M_MF_B, M_RVC_B, M_MF_A, M_RVC_A, M_MF_D, M_RVC_D],
        "targets": ["Waist Comfort", "Waist Compression", "Waist Sore Swelling"],
    },
}
GLOBAL = {
    "feats": ANTHRO + [IMPACT, L4L5, DL4L5, T_MF_B, T_RVC_B, T_MF_A, T_RVC_A, T_MF_D, T_RVC_D,
                       E_MF_B, E_RVC_B, E_MF_A, E_RVC_A, E_MF_D, E_RVC_D,
                       M_MF_B, M_RVC_B, M_MF_A, M_RVC_A, M_MF_D, M_RVC_D],
    "target": "Comprehensive comfort",
}
BASELINES = {
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42),
}


def present(columns, wanted):
    return [c for c in wanted if c in columns]


def load_region(name):
    cfg = REGION[name]
    df = pd.read_excel(DATA, sheet_name=cfg["sheet"])
    feats = present(df.columns, cfg["feats"])
    bp = pd.get_dummies(df["backpack"], prefix="bp").to_numpy(float)
    X = np.hstack([df[feats].to_numpy(float), bp])
    Y = df[cfg["targets"]].to_numpy(float)
    names = feats + [f"Bpk{i + 1}" for i in range(bp.shape[1])]
    return X, Y, df["participant_id"].to_numpy(), df["fold"].to_numpy(), names, cfg["targets"]


def load_global():
    df = pd.read_excel(DATA, sheet_name="global")
    feats = present(df.columns, GLOBAL["feats"])
    bp = pd.get_dummies(df["backpack"], prefix="bp").to_numpy(float)
    X = np.hstack([df[feats].to_numpy(float), bp])
    Y = df[[GLOBAL["target"]]].to_numpy(float)
    names = feats + [f"Bpk{i + 1}" for i in range(bp.shape[1])]
    return X, Y, df["participant_id"].to_numpy(), df["fold"].to_numpy(), names


def center(X, Y, g):
    Xc = X.copy()
    Yc = Y.copy()
    for s in np.unique(g):
        m = g == s
        Xc[m] = X[m] - X[m].mean(0)
        Yc[m] = Y[m] - Y[m].mean(0)
    return Xc, Yc


def pca_for(Xtr):
    n = Xtr.shape[0]
    probe = PCA(n_components=min(Xtr.shape[1], n - 1), svd_solver="full").fit(Xtr)
    cum = np.cumsum(probe.explained_variance_ratio_)
    k = int(np.searchsorted(cum, PCA_VAR) + 1)
    k = max(2, min(k, Xtr.shape[1], n - 1))
    return PCA(n_components=k, svd_solver="full").fit(Xtr)


def fit_fold(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    A = scaler.transform(Xtr)
    B = scaler.transform(Xte)
    pca = pca_for(A)
    Pa = pca.transform(A)
    Pb = pca.transform(B)
    model = xgb.train(XGB_PARAMS, xgb.DMatrix(Pa, label=ytr), num_boost_round=N_ROUNDS)
    return model, pca, Pa, Pb


def scores(y, p):
    r2 = float(r2_score(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan
    return r2, rmse, mae, r


def summarize(rows, keys):
    out = {}
    for k in ("r2", "rmse", "mae", "r"):
        vals = np.array([row[k] for row in rows], float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals))
    out.update({key: rows[0][key] for key in keys})
    return out


def analyse(name, X, Y, pid, fold, targets):
    Xc, Yc = center(X, Y, pid)
    pred_rows, fold_rows, base_rows, shap_rows = [], [], [], []
    for t, target in enumerate(targets):
        per_fold = []
        for f in range(1, K + 1):
            tr = fold != f
            te = fold == f
            model, pca, Pa, Pb = fit_fold(Xc[tr], Yc[tr, t], Xc[te])
            pred = model.predict(xgb.DMatrix(Pb))
            r2, rmse, mae, r = scores(Yc[te, t], pred)
            per_fold.append({"region": name, "attribute": target, "fold": f,
                             "n_test": int(te.sum()), "r2": r2, "rmse": rmse, "mae": mae, "r": r})
            vals = shap.TreeExplainer(model).shap_values(Pb)
            for j in range(vals.shape[1]):
                shap_rows.append({"region": name, "attribute": target, "fold": f,
                                  "component": j + 1,
                                  "mean_abs_shap": float(np.mean(np.abs(vals[:, j])))})
        fold_rows.extend(per_fold)
        pred_rows.append(summarize(per_fold, ("region", "attribute")))

        for model_name, estimator in BASELINES.items():
            base_fold = []
            for f in range(1, K + 1):
                tr = fold != f
                te = fold == f
                scaler = StandardScaler().fit(Xc[tr])
                A = scaler.transform(Xc[tr])
                B = scaler.transform(Xc[te])
                pca = pca_for(A)
                Pa = pca.transform(A)
                Pb = pca.transform(B)
                estimator.fit(Pa, Yc[tr, t])
                p = estimator.predict(Pb)
                base_fold.append({"r2": float(r2_score(Yc[te, t], p)),
                                  "rmse": float(np.sqrt(mean_squared_error(Yc[te, t], p)))})
            base_rows.append({"region": name, "attribute": target, "model": model_name,
                              "r2_mean": float(np.mean([b["r2"] for b in base_fold])),
                              "r2_sd": float(np.std([b["r2"] for b in base_fold])),
                              "rmse_mean": float(np.mean([b["rmse"] for b in base_fold])),
                              "rmse_sd": float(np.std([b["rmse"] for b in base_fold]))})
    return pred_rows, fold_rows, base_rows, shap_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pred, folds, bases, shaps = [], [], [], []
    for name in REGION:
        X, Y, pid, fold, names, targets = load_region(name)
        a, b, c, d = analyse(name, X, Y, pid, fold, targets)
        pred += a
        folds += b
        bases += c
        shaps += d

    X, Y, pid, fold, names = load_global()
    a, b, c, d = analyse("Overall", X, Y, pid, fold, [GLOBAL["target"]])
    for row in a:
        row["attribute"] = "Comfort"
    for row in b:
        row["attribute"] = "Comfort"
    for row in c:
        row["attribute"] = "Comfort"
    for row in d:
        row["attribute"] = "Comfort"
    pred += a
    folds += b
    bases += c
    shaps += d

    pd.DataFrame(pred)[["region", "attribute", "r2_mean", "r2_sd", "rmse_mean", "rmse_sd",
                        "mae_mean", "mae_sd", "r_mean", "r_sd"]].to_csv(
        os.path.join(OUT, "table4_prediction.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(OUT, "tableS10_per_fold.csv"), index=False)
    pd.DataFrame(bases).to_csv(os.path.join(OUT, "tableS15_baselines.csv"), index=False)
    pd.DataFrame(shaps).to_csv(os.path.join(OUT, "tableS11_shap.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rounds": N_ROUNDS, "pca_variance": PCA_VAR, "folds": K,
                   "anthropometry_columns_used": [c for c in ANTHRO if c in pd.read_excel(
                       DATA, sheet_name="global").columns],
                   "prediction": pred}, f, indent=2)
    print(pd.DataFrame(pred)[["region", "attribute", "r2_mean", "r2_sd", "mae_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
