# Disease Prediction from Medical Data

Predict the likelihood of **heart disease, diabetes and breast cancer** from structured patient data using classical ML classifiers.

## Datasets (UCI ML Repository)
| Dataset | Source | Target |
|---|---|---|
| Heart Disease (Cleveland) | UCI | 1 = disease present |
| Pima Indians Diabetes | UCI / NIDDK | 1 = diabetic |
| Breast Cancer Wisconsin | UCI (bundled in scikit-learn) | 1 = malignant |

## Algorithms
Logistic Regression, SVM (RBF), Random Forest, XGBoost.

## Pipeline (step by step)
1. **Load data** - heart & diabetes are downloaded once into `data/`; cancer is built into scikit-learn.
2. **Clean** - `?` in heart data and impossible `0` values in diabetes (glucose, BMI, ...) become missing values.
3. **Preprocess** - median imputation + standard scaling inside a scikit-learn `Pipeline` (prevents data leakage).
4. **Split** - 80/20 stratified train/test split.
5. **Train** - 4 models, with 5-fold stratified cross-validation (ROC-AUC) on the training set.
6. **Evaluate** - Accuracy, Precision, Recall, F1, ROC-AUC on the held-out test set.
7. **Visualise** - ROC curves, confusion matrix of best model, top Random Forest feature importances.
8. **Save** - best model per dataset as `.joblib`, results as `.csv`.

## Setup & Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python disease_prediction.py                  # all datasets
python disease_prediction.py --dataset heart  # one dataset: heart | diabetes | cancer
```
Needs internet on the first run for the heart and diabetes files. Or place `heart.csv` / `diabetes.csv` in `data/` manually.

## Output (`outputs/`)
`*_results.csv`, `*_roc.png`, `*_confusion_matrix.png`, `*_feature_importance.png`, `*_best_model.joblib`

## Using a saved model
```python
import joblib, pandas as pd
model = joblib.load("outputs/cancer_best_model.joblib")
print(model.predict_proba(new_patient_dataframe)[:, 1])  # probability of disease
```

## Notes
- In medicine, **Recall** (not missing sick patients) often matters more than accuracy - compare it across models.
- Datasets are small, so results vary a little by split; cross-validation scores are more stable.
- This is an educational project, **not** a diagnostic tool.
