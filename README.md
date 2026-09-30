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
6. **Deployment** — metric recommendations based on error cost in production.

> **Why PR-AUC and not accuracy?** With only 3% fraud, a trivial model reaches 97%
> accuracy without detecting any fraud. The precision-recall curve accounts for the
> minority class and aligns the metric with the real cost of the business.

## Repository structure

```
├── content/
│   └── credit-card-fraud.csv      # Dataset (5,000 transactions)
├── notebooks/
│   └── main.ipynb                 # Complete CRISP-DM project
├── pyproject.toml                 # Dependencies (Poetry)
├── trabajo-final.docx             # Course delivery document (Spanish)
└── LICENSE
```

## Requirements

- Python 3.11–3.13
- [Poetry](https://python-poetry.org/) (optional, see `pyproject.toml`)
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`,
  `seaborn`, `ydata-profiling`

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

## Tech stack

- **scikit-learn** — models, `GridSearchCV`, `StratifiedKFold` and metrics
- **imbalanced-learn** — `RandomUnderSampler` and `SMOTE`
- **pandas / numpy** — processing and analysis
- **matplotlib / seaborn** — visualization
- **ydata-profiling** — exploratory profiling

## License

[MIT](LICENSE)