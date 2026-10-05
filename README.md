# Credit card fraud detection in banking transactions using ML

Machine Learning project that classifies credit card transactions as **fraudulent** or
**legitimate** following the **CRISP-DM** methodology. Developed as the final project of
the Machine Learning course.

**Authors:** Jacobo Blandón Castro · Johan Antonio Peña López

## Description

Undetected fraud causes financial losses and undermines trust in payment methods. The
goal is to automatically decide, at authorization time, whether a transaction is
fraudulent based on its features (amount, time, merchant, and anonymized components
derived from an original PCA).

The working dataset is **synthetic**: 5,000 transactions, 22 features and a target
variable `Class` that is imbalanced (3% fraud rate), similar to real-world datasets in
this domain.

## Results

The best model is **Logistic Regression**, trained with SMOTE balancing and evaluated on
an unbalanced test set (true fraud rate):

| Metric | Value |
|---|---|
| PR-AUC | **0.9981** |
| ROC-AUC | **0.9999** |
| Recall | **100%** (45/45 frauds detected) |
| Precision | 87% |
| F1 | 0.9278 |

**Eight models** were compared (five classical and three ensembles) with hyperparameter
tuning via `GridSearchCV` + 5-fold `StratifiedKFold` on 70% of the data:

| Model | PR-AUC (test) |
|---|---|
| Logistic Regression | **0.9981** |
| SVM | 0.9943 |
| Voting (*soft voting*) | 0.9923 |
| Random Forest (*bagging*) | 0.9920 |
| Gradient Boosting (*boosting*) | 0.9858 |
| Naive Bayes | 0.9287 |
| KNN | 0.8923 |
| Decision Tree | 0.7399 |

## CRISP-DM methodology

The notebook `notebooks/main.ipynb` documents the six phases:

1. **Business understanding** — problem, objective and definition of the target class.
2. **Data understanding** — data dictionary, quality rules and missing-value analysis
   (MCAR).
3. **Data preparation** — feature selection based on business rules, missing-value
   imputation, `Amount` reconstruction from `Amount_log`, one-hot encoding,
   standardization, correlation analysis and diagnostic PCA.
4. **Modeling** — class balancing (SMOTE), five classical models and three ensembles,
   tuned with cross-validation.
5. **Evaluation** — quality metrics on the test set and best-model selection.
6. **Deployment** — metric recommendations based on error cost in production, and the
   Streamlit app in `app.py` that applies them.

> **Why PR-AUC and not accuracy?** With only 3% fraud, a trivial model reaches 97%
> accuracy without detecting any fraud. The precision-recall curve accounts for the
> minority class and aligns the metric with the real cost of the business.

## Repository structure

```
├── app.py                          # Streamlit app (deployment of the model)
├── entrenar_modelo.py              # Trains the model and exports the artifacts
├── preprocesamiento.py             # Feature engineering shared by training and app
├── content/
│   └── credit-card-fraud.csv      # Dataset (5,000 transactions)
├── modelos/                        # Artifacts consumed by the app
│   ├── modelo_fraude.joblib       # Logistic regression (SMOTE)
│   ├── preprocesador.joblib       # StandardScaler + OneHotEncoding
│   ├── metadatos.joblib           # Medians, categories and test metrics
│   └── plantilla_transacciones.xlsx
├── notebooks/
│   └── main.ipynb                 # Complete CRISP-DM project
├── pyproject.toml                 # Dependencies (Poetry)
├── requirements.txt               # Dependencies (pip, for the app)
├── trabajo-final.docx             # Course delivery document (Spanish)
└── LICENSE
```

## Requirements

- Python 3.11–3.13
- [Poetry](https://python-poetry.org/) (optional, see `pyproject.toml`)
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`,
  `seaborn`, `ydata-profiling`, `streamlit`, `openpyxl`, `joblib`

## Usage

### Google Colab (recommended)

1. Open the notebook `notebooks/main.ipynb`.
2. In Colab the dataset is **downloaded automatically** from this repository when the
   loading cell runs; no manual upload to `/content` is required.
3. Run the cells in order (`Runtime → Run all`). Installing dependencies takes a few
   seconds and the hyperparameter search about a minute.

### Locally

```bash
poetry install
poetry run python -m ipykernel install --user --name credit-card-fraud
poetry run jupyter lab
# open notebooks/main.ipynb and run the cells in order
```

The loading cell uses the local path `../content/credit-card-fraud.csv` and, if the file
does not exist, also downloads it from the repository.

## App deployment (Streamlit)

`app.py` deploys the final model: it receives a transaction and returns the probability of
fraud and the decision (approve or block) taken at authorization time.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Two input paths are available:

- **Transacción individual** — form with the 13 model inputs (`V1`–`V10`, `Time`, `Amount`,
  `MerchantCategory`), showing the processed features and, per transaction, the contribution
  of each variable to the decision.
- **Lote de transacciones** — Excel/CSV upload scored in batch. If the file includes the
  `Class` column, the app reports recall, precision and the confusion matrix for that file.
  A ready-to-use template is available for download inside the app.

The sidebar exposes the metrics measured on the test set and the decision threshold, which
trades recall against false positives.

### Retraining the artifacts

```bash
poetry run python entrenar_modelo.py     # or: python entrenar_modelo.py
```

The script repeats the notebook pipeline (cleaning, `ColumnTransformer`, SMOTE,
`GridSearchCV`) and writes `modelos/*.joblib`. At the end it reloads the artifacts from disk
and verifies that they reproduce the test metrics of the notebook.

> `preprocesamiento.py` is the single source of truth for feature engineering, so the app can
> never diverge from training. `Amount` is always rebuilt from `Amount_log` when missing or
> inconsistent, `V1`/`V3` are imputed with the training medians and their missing-value flags
> are rebuilt from the input, and `Hora`/`Franja` are derived from `Time`.

## Tech stack

- **scikit-learn** — models, `GridSearchCV`, `StratifiedKFold` and metrics
- **imbalanced-learn** — `RandomUnderSampler` and `SMOTE`
- **pandas / numpy** — processing and analysis
- **matplotlib / seaborn** — visualization
- **ydata-profiling** — exploratory profiling
- **Streamlit / joblib** — deployment of the model and its artifacts

## License

[MIT](LICENSE)