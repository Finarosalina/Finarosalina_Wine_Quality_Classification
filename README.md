# 🍷 Wine Quality Classification — KNN Model

**Multiclass classification of red wine quality using K-Nearest Neighbors, with class balancing and hyperparameter optimization.**

---

## Overview

This project builds a machine learning pipeline to classify red wine quality into three categories (low, medium, high) using physicochemical features. The dataset contains 1,599 samples with 11 input features and a quality score from 3 to 8.

The project explores two labeling strategies, compares their performance, and optimizes the final model using GridSearchCV and undersampling techniques to handle class imbalance.

---

## Problem Definition

Wine quality scores (3–8) are mapped to three classes:

| Class | Label | Quality range |
|---|---|---|
| 0 | Low quality | ≤ 5 |
| 1 | Medium quality | 6 |
| 2 | High quality | ≥ 7 |

Two labeling schemes were evaluated before settling on the most balanced approach.

---

## Methodology

**1. Exploratory Data Analysis**
- Distribution analysis of quality scores
- Feature statistics and correlation review
- Identification of class imbalance (class 2 underrepresented)

**2. Labeling strategies**
- Scheme 1: low (≤4) / medium (5–6) / high (≥7) — heavily imbalanced
- Scheme 2: low (≤5) / medium (6) / high (≥7) — more balanced, selected as final

**3. Preprocessing**
- StandardScaler normalization
- Stratified train/test split (70/30)

**4. Model optimization**
- GridSearchCV over k ∈ [13, 17] with F1-weighted scoring and 5-fold cross-validation
- Best k found: 14
- Additional test with distance-weighted KNN + RandomUnderSampler

**5. Evaluation**
- Accuracy, F1-score (weighted and macro), Balanced Accuracy
- Confusion matrix visualization
- Full classification report per class

---

## Results

| Model | Accuracy | F1 Weighted | Balanced Accuracy |
|---|---|---|---|
| KNN k=14 (GridSearchCV) | 0.644 | 0.640 | — |
| KNN k=13 weighted + undersampling | 0.615 | 0.618 | 0.666 |
| KNN k=1 (baseline scheme 2) | 0.675 | 0.680 | — |

The baseline scheme 2 (k=1) achieved the best global accuracy, while the undersampled model improved recall for the minority class (high quality wines).

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, imbalanced-learn
- **Visualization** — matplotlib, seaborn
- **Model persistence** — joblib
- **Environment** — Jupyter Notebook / VS Code

---

## Project Structure

```
Finarosalina_KNN_BUENO_ML_WINE/
├── src/
│   ├── explore.ipynb       # Full analysis and model development
│   └── app.py              # Exported pipeline script
├── models/
│   ├── final_model.pkl     # GridSearchCV optimized model
│   └── knn_model.pkl       # Distance-weighted undersampled model
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Finarosalina/Finarosalina_KNN_BUENO_ML_WINE.git
cd Finarosalina_KNN_BUENO_ML_WINE

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

# Run the notebook
jupyter notebook src/explore.ipynb
```

---

## Key Learnings

- Class imbalance significantly affects KNN performance — undersampling improved recall for minority classes at the cost of overall accuracy
- The choice of labeling scheme is a domain decision, not just a technical one
- GridSearchCV with stratified cross-validation is essential for reliable k selection in imbalanced datasets
- Distance-weighted KNN helps partially compensate for imbalance without data augmentation

---

## Dataset

[Red Wine Quality — UCI / 4Geeks Academy](https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv)

---

*Part of the 4Geeks Academy Data Science & ML program portfolio.*
