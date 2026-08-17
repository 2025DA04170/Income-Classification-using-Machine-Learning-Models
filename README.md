# Machine Learning Assignment 2 - Income Classification using Adult Dataset

## a. Problem Statement

The objective of this project is to build, evaluate, and deploy multiple machine learning classification models for predicting whether a person's income is less than or equal to 50K or greater than 50K based on demographic, employment-related and other attributes.

## b. Dataset Description

- **Dataset Name:** Adult Census Income Dataset
- **Source:** UCI Machine Learning Dataset Repository ([link](https://archive.ics.uci.edu/dataset/2/adult))
- **Task Type:** Binary Classification
- **Number of Instances:** 32,561 (after removing rows with a missing target)
- **Number of Features:** 14 (6 numeric, 8 categorical)
- **Target Variable:** `income`
- **Classes:** `<=50K` and `>50K`

**Features used:** age, workclass, fnlwgt, education, education.num, marital.status, occupation, relationship, race, sex, capital.gain, capital.loss, hours.per.week, native.country.

**Preprocessing:**
- `?` values in categorical columns were treated as missing.
- Missing numeric values were imputed with the median; missing categorical values with the most frequent category.
- Numeric features were standardized with `StandardScaler`.
- Categorical features were encoded with `OneHotEncoder` (unknown categories at inference time are safely ignored).
- The target was label-encoded (`<=50K` → 0, `>50K` → 1).
- Data was split 80/20 (train/test) with stratification on the target.

## c. GitHub Repository Link

GitHub Repository: ([Income-Classification-using-Machine-Learning-Models](https://github.com/2025DA04170/Income-Classification-using-Machine-Learning-Models.git))

## d. Models Used

The following classification models were implemented on identical train/test splits:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8531 | 0.9024 | 0.7392 | 0.6020 | 0.6636 | 0.5758 |
| Decision Tree | 0.8153 | 0.7521 | 0.6133 | 0.6301 | 0.6216 | 0.4995 |
| kNN | 0.8406 | 0.8869 | 0.6931 | 0.6065 | 0.6469 | 0.5466 |
| Naive Bayes | 0.5366 | 0.7334 | 0.3360 | 0.9471 | 0.4960 | 0.3237 |
| Random Forest (Ensemble) | 0.8597 | 0.9113 | 0.7904 | 0.5676 | 0.6607 | 0.5878 |

*(Full precision values are in `model/metrics.csv & model/metrics.json`, regenerated automatically each time `train_models.py` is run.)*

## Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced performer — highest AUC alongside Random Forest, with good precision. As a linear model it can't capture complex feature interactions, but the one-hot/standardized features and the roughly linear relationship between income and factors like education and hours worked suit it well. |
| Decision Tree | Weakest of the tree-based/ensemble models — a single unpruned tree overfits the training data, which shows up as the lowest AUC (0.75) despite reasonable accuracy. Recall is its best metric, but precision and MCC lag behind the ensemble. |
| kNN | Solid mid-table performance. Distance-based classification benefits from the scaled numeric features but is diluted by the high-dimensional one-hot-encoded categorical space, where "nearness" becomes less meaningful (the curse of dimensionality). |
| Naive Bayes | Clearly the weakest model on this dataset. Its independence assumption is badly violated by correlated features (e.g., education vs. education.num, or occupation vs. hours.per.week), and GaussianNB assumes normality even for the one-hot encoded categorical columns. It does achieve the highest recall (0.947) because it aggressively predicts the >50K class, but at the cost of very low precision and accuracy near chance level for a class-imbalanced problem. |
| Random Forest (Ensemble) | Best overall model — highest accuracy, AUC, precision, and MCC. Averaging many depth-limited trees fixes the single tree's overfitting problem while still capturing non-linear interactions between features that Logistic Regression misses. Precision is notably higher than every other model, though recall is slightly lower, meaning it is more conservative about predicting the `>50K` class. |
| **Overall Winner** | **Random Forest (Ensemble)** — it has the best MCC (0.588), AUC (0.911), and accuracy (0.860) of all five models, and comes with the best precision by a clear margin. Logistic Regression is a strong, more interpretable runner-up with the best recall among the two strongest models and near-identical AUC. |

## Streamlit App Link

Live App: ([Income Classification · ML Models](https://income-classification-using-machine-learning-models.streamlit.app/))

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the models from scratch:

```bash
cd model
python train_models.py
```

This regenerates `adult_cleaned.csv`, `test_data.csv`, all five `*_model.pkl` files, `preprocessor.pkl`, and `model/metrics.csv`.

## Repository Contents

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `data/raw/adult_income_raw.csv` — original dataset (downloaded from UCI repository)
- `data/processed/test_data.csv` — held-out test split (features + true label) for demo
- `data/processed/adult_cleaned.csv` — cleaned full dataset
- `model/` — training script, saved model `.pkl` files, preprocessor, and metrics
- `notebook/` — Jupyter notebook with the full implementation and exploration
- `assets/` — static assets for the Streamlit app (favicon)
