"""
Disease Prediction from Medical Data
====================================
Datasets : Heart Disease (Cleveland), Pima Diabetes, Breast Cancer Wisconsin (UCI)
Models   : Logistic Regression, SVM, Random Forest, XGBoost

Usage:
    python disease_prediction.py                 # run all 3 datasets
    python disease_prediction.py --dataset heart # run one (heart | diabetes | cancer)
"""
import argparse
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
DATA_DIR, OUT_DIR = "data", "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 1: Load datasets
# --------------------------------------------------------------------------
def load_heart():
    """UCI Heart Disease (Cleveland). Target: 0 = no disease, 1 = disease."""
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"]
    path = os.path.join(DATA_DIR, "heart.csv")
    if not os.path.exists(path):
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        pd.read_csv(url, names=cols, na_values="?").to_csv(path, index=False)
    df = pd.read_csv(path)
    df["target"] = (df["target"] > 0).astype(int)  # 1-4 -> 1
    return df.drop(columns="target"), df["target"]


def load_diabetes():
    """Pima Indians Diabetes. Target: 1 = diabetic."""
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "pedigree", "age", "target"]
    path = os.path.join(DATA_DIR, "diabetes.csv")
    if not os.path.exists(path):
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
               "master/pima-indians-diabetes.data.csv")
        pd.read_csv(url, names=cols).to_csv(path, index=False)
    df = pd.read_csv(path)
    # In this dataset, 0 is impossible for these columns -> it means "missing"
    for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df[c] = df[c].replace(0, np.nan)
    return df.drop(columns="target"), df["target"]


def load_cancer():
    """Breast Cancer Wisconsin (built into scikit-learn). Target: 1 = malignant."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = 1 - data.target  # sklearn: 0 = malignant, so flip so 1 = disease
    return X, y


DATASETS = {"heart": load_heart, "diabetes": load_diabetes, "cancer": load_cancer}


# --------------------------------------------------------------------------
# STEP 2: Models (preprocessing + classifier in one Pipeline -> no data leakage)
# --------------------------------------------------------------------------
def build_models(X):
    def pipe(clf):
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), list(X.columns))
        ])
        return Pipeline([("pre", pre), ("clf", clf)])

    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, random_state=SEED)),
        "SVM": pipe(SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED)),
        "XGBoost": pipe(XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      eval_metric="logloss", random_state=SEED)),
    }


# --------------------------------------------------------------------------
# STEP 3-5: Train, evaluate, visualise, save
# --------------------------------------------------------------------------
def run(name):
    print(f"\n{'=' * 60}\nDATASET: {name.upper()}\n{'=' * 60}")
    X, y = DATASETS[name]()
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Positive rate: {y.mean():.2%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for model_name, model in build_models(X).items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        model.fit(X_tr, y_tr)
        pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
        rows.append({
            "Model": model_name,
            "CV ROC-AUC": round(cv_auc, 4),
            "Accuracy": round(accuracy_score(y_te, pred), 4),
            "Precision": round(precision_score(y_te, pred), 4),
            "Recall": round(recall_score(y_te, pred), 4),
            "F1": round(f1_score(y_te, pred), 4),
            "ROC-AUC": round(roc_auc_score(y_te, proba), 4),
        })
        fitted[model_name] = model
        RocCurveDisplay.from_predictions(y_te, proba, name=model_name, ax=ax_roc)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title(f"ROC Curves - {name}")
    fig_roc.tight_layout()
    fig_roc.savefig(f"{OUT_DIR}/{name}_roc.png", dpi=150)
    plt.close(fig_roc)

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    print(results.to_string(index=False))
    results.to_csv(f"{OUT_DIR}/{name}_results.csv", index=False)

    # Best model (by test ROC-AUC): confusion matrix + save
    best_name = results.iloc[0]["Model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    ConfusionMatrixDisplay.from_estimator(
        best, X_te, y_te, display_labels=["Healthy", "Disease"], cmap="Blues")
    plt.title(f"{best_name} - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}_confusion_matrix.png", dpi=150)
    plt.close()
    joblib.dump(best, f"{OUT_DIR}/{name}_best_model.joblib")

    # Feature importance (Random Forest) - useful for explaining the result
    rf = fitted["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
    imp.plot(kind="barh", figsize=(7, 5), title=f"Top features (Random Forest) - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}_feature_importance.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    args = parser.parse_args()
    for name in (DATASETS if args.dataset == "all" else [args.dataset]):
        run(name)
    print(f"\nDone. Results, plots and models saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
