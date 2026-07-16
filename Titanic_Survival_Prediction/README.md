# Titanic Survival Prediction Using Decision Tree Classifier

## Project Overview

**Problem Statement**
The sinking of the Titanic resulted in the loss of a large number of passengers. Historical passenger data records information such as age, sex, ticket class, fare, and family relationships. This project uses that data to predict whether a given passenger survived.

**Project Objective**
Build a supervised machine learning classification model that predicts passenger survival (`Survived`: 0 = Did not survive, 1 = Survived) based on passenger attributes.

**Why This Machine Learning Approach Was Selected**
A Decision Tree Classifier was selected because:
- The target variable is binary (survived / did not survive).
- Decision Trees handle a mix of numerical and categorical features well.
- Decision Trees do not require feature scaling.
- Decision Trees are interpretable, allowing direct visualization of the decision-making process and feature importance.

**Expected Outcome**
A trained classification model capable of predicting survival outcomes on unseen passenger data, along with a quantitative evaluation of its performance.

---

## Dataset Information

| Attribute | Detail |
|---|---|
| Dataset Name | Titanic-Dataset.csv |
| Dataset Source | Not Available (file path referenced in notebook: `../Data/Titanic-Dataset.csv`) |
| Dataset Size | 891 rows x 12 columns |
| Target Variable | `Survived` |

**Original Features**

| Column | Description | Non-Null Count |
|---|---|---|
| PassengerId | Unique passenger identifier | 891 |
| Survived | Target variable (0 = No, 1 = Yes) | 891 |
| Pclass | Ticket class (1st, 2nd, 3rd) | 891 |
| Name | Passenger name | 891 |
| Sex | Passenger gender | 891 |
| Age | Passenger age | 714 |
| SibSp | Number of siblings/spouses aboard | 891 |
| Parch | Number of parents/children aboard | 891 |
| Ticket | Ticket number | 891 |
| Fare | Ticket fare | 891 |
| Cabin | Cabin number | 204 |
| Embarked | Port of embarkation | 889 |

**Missing Values Summary**

| Column | Missing Values |
|---|---|
| Age | 177 |
| Cabin | 687 (more than 77% of records) |
| Embarked | 2 |

---

## Project Workflow

1. Import required libraries
2. Load the dataset
3. Understand the dataset (shape, data types, summary statistics, column names)
4. Data cleaning (duplicate check, missing value inspection and treatment)
5. Exploratory Data Analysis (target distribution, numerical distribution, categorical distribution, correlation heatmap)
6. Feature selection (dropping non-predictive columns)
7. Feature encoding (converting categorical columns to numerical form)
8. Define features (`x`) and target (`y`)
9. Train-test split
10. Build the Decision Tree model
11. Make predictions
12. Evaluate the model (accuracy, precision, recall, F1-score, confusion matrix)
13. Feature importance analysis
14. Classification report
15. Cross-validation
16. Save the trained model

---

## Exploratory Data Analysis (EDA)

- **Target Variable Distribution**: `Survived` value counts — 549 passengers did not survive, 342 survived.
- **Numerical Feature Distribution**: Histogram with KDE for `Age` and `Fare`.
<p align="center">
  <img src="images/Survival Distribution.png" width="600">
</p>
- **Categorical Feature Distribution**: Count plots for `Sex`, `Pclass`, and `Embarked`.
<p align="center">
  <img src="images/Categorical Features Distribution.png" width="600">
</p>
- **Skewness Analysis**: `Age` skewness calculated as approximately 0.389; `Fare` skewness calculated as approximately 4.787 (highly right-skewed).
- **Correlation Analysis**: Correlation heatmap generated on numeric features.

Boxplots, outlier treatment, and ROC-AUC analysis were not present in the notebook.

---

## Data Preprocessing

**Duplicate Removal**
Checked for duplicate rows using `df.duplicated().sum()`.

**Missing Value Handling**
- `Age`: missing values filled with the median.
- `Embarked`: missing values filled with the mode.
- `Cabin`: column dropped due to more than 77% missing values (not imputed).

**Feature Selection**
The following columns were removed:
- `PassengerId` — unique identifier, not useful for prediction.
- `Name` — high-cardinality text feature.
- `Ticket` — high-cardinality text feature.
- `Cabin` — more than 77% missing values.

**Feature Encoding**
- `Sex` mapped to numerical values: male = 0, female = 1.
- `Embarked` mapped to numerical values: S = 0, C = 1, Q = 2.

**Feature Scaling**
Not applied. Decision Tree models do not require feature scaling.

**Outlier Treatment**
Not performed in the notebook.

---

## Model Development

**Algorithm Used**
`DecisionTreeClassifier` from `scikit-learn`, with `random_state=42`.

**Why This Algorithm Was Selected**
Selected for its interpretability, ability to handle categorical and numerical data without scaling, and suitability for binary classification tasks.

**Model Training Process**
- Features (`x`): `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
- Target (`y`): `Survived`.
- Data split into training and test sets using `train_test_split` with `test_size=0.25` and `random_state=2`.
- Training set size: 668 samples. Test set size: 223 samples.
- Model trained using `model.fit(x_train, y_train)`.

**Training Pipeline**
A `scikit-learn` `Pipeline` containing the `DecisionTreeClassifier` was used specifically for cross-validation.

---

## Model Evaluation

| Metric | Score |
|---|---|
| Accuracy | 0.77 |
| Precision | 0.73 |
| Recall | 0.70 |
| F1-Score | 0.71 |

**Confusion Matrix**
Generated and visualized using a Seaborn heatmap.

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.79      0.82      0.80       131
           1       0.73      0.70      0.71        92

    accuracy                           0.77       223
   macro avg       0.76      0.76      0.76       223
weighted avg       0.77      0.77      0.77       223
```

**Cross-Validation**
5-fold cross-validation performed using a Pipeline wrapping the Decision Tree model.
Average cross-validation accuracy: 0.7778

ROC-AUC, MAE, MSE, RMSE, and R² Score were not applicable/not computed, as this is a classification task evaluated with the metrics above.

---

## Feature Importance

Feature importance was computed using `model.feature_importances_` and visualized as a horizontal bar chart.

- **What it represents**: The relative contribution of each feature to the Decision Tree's splitting decisions and, consequently, to its predictions.
- **Key observations**: The chart ranks all seven input features (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`) by importance score. Exact ranking values are available in the notebook output.
- **Why it is useful**: Helps identify which passenger attributes most strongly influence survival predictions, supporting model interpretability.

---

## Model Visualization

- **Decision Tree Plot**: Visualized using `plot_tree` (scikit-learn), limited to a depth of 3 for readability, with feature names and class labels (`Died`, `Survived`).
- **Feature Importance Plot**: Horizontal bar chart of feature importance scores.
- **Confusion Matrix**: Seaborn heatmap with annotated counts.
- **Correlation Heatmap**: Seaborn heatmap of numeric feature correlations.
- **Distribution Plots**: Histograms with KDE for `Age` and `Fare`; count plots for `Survived`, `Sex`, `Pclass`, and `Embarked`.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Data visualization |
| seaborn | Statistical data visualization |
| scikit-learn | Model training, pipeline, evaluation, cross-validation |
| joblib | Model serialization |

---

## Project Structure

```
Titanic_Survival_Prediction/
├── Data/
│   └── Titanic-Dataset.csv
├── Notebooks/
│   └── Titanic_Survival_Decision_Tree.ipynb
├── Model/
│   └── Decision_tree_model.pkl
└── README.md
```

Note: Folder structure inferred from relative file paths used in the notebook (`../Data/...`, `../Model/...`).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/junaidrehman41/ML_Projects.git
   cd ML_Projects/Titanic_Survival_Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Open the notebook located at `Notebooks/Titanic_Survival_Decision_Tree.ipynb`.
2. Run all cells sequentially to reproduce data cleaning, EDA, model training, and evaluation.
3. The trained model is saved to `Model/Decision_tree_model.pkl` for reuse.

---

## Results

The Decision Tree Classifier achieved 77% accuracy on the test set, with a precision of 0.73, recall of 0.70, and F1-score of 0.71 for the positive (survived) class. Five-fold cross-validation confirmed model stability with an average accuracy of 0.778, closely matching the single train-test split result.

---

## Learning Outcomes

This project demonstrates:
- End-to-end classification workflow: data cleaning, EDA, preprocessing, modeling, and evaluation.
- Handling of missing data using median and mode imputation.
- Categorical feature encoding for machine learning compatibility.
- Training and evaluating a Decision Tree Classifier.
- Model interpretability through decision tree visualization and feature importance analysis.
- Model validation using k-fold cross-validation.
- Model persistence for later reuse or deployment.

---

## Future Improvements

- Apply `stratify=y` in the train-test split to preserve class distribution between training and test sets.
- Tune hyperparameters such as `max_depth` and `min_samples_split` to reduce overfitting risk.
- Compare performance against ensemble methods such as Random Forest.
- Incorporate ROC-AUC analysis for a threshold-independent performance view.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## Author

- Name: Junaid Ur Rehman
- GitHub: https://github.com/junaidrehman41
- Email: junaidrehman41@gmail.com