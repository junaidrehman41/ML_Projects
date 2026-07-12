# 🎓 Student Exam Score Prediction — Logistic Regression

A machine learning project that predicts whether a student will **Pass** or **Fail** an exam based on study habits and academic history, using Logistic Regression. The project covers the full pipeline — from raw data to a deployed, interactive web app.

---

## 📌 Project Overview

Given a student's study and lifestyle metrics, the model predicts a binary outcome: **Pass (1)** or **Fail (0)**. A student is labeled as *Pass* if their exam score is **30 or above** (out of a total of 60).

---

## 📂 Dataset

The dataset (`Data/student_exam_scores.csv`) contains **200 student records** with the following columns:

| Column                | Description                              |
|-----------------------|-------------------------------------------|
| `student_id`          | Unique student identifier                 |
| `hours_studied`       | Number of hours studied                   |
| `sleep_hours`         | Average hours of sleep                    |
| `attendance_percent`  | Class attendance percentage               |
| `previous_scores`     | Score from a previous exam                |
| `exam_score`          | Final exam score (used to derive the target) |

**Target variable:** `pass_fail` — engineered from `exam_score` using the rule `exam_score >= 30 → 1 (Pass), else 0 (Fail)`.

---

## 🗂️ Project Structure

```
Student_Exam_Scores_Prediction/
├── Data/
│   └── student_exam_scores.csv
├── Model/
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl
├── Notebooks/
│   └── Student_Performance_Logistic_Regression.ipynb
├── ScreenShots/
├── app.py
├── requirements.txt
├── .gitignore
└── Readme.md
```

---

## 🔎 Methodology

The notebook (`Notebooks/Student_Performance_Logistic_Regression.ipynb`) follows a structured, end-to-end workflow:

1. **Import Libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn
2. **Load Dataset** — read the CSV into a DataFrame
3. **Understand the Dataset** — shape, info, summary statistics
4. **Data Cleaning** — checked for duplicates and missing values (none found)
5. **Create Target Variable** — derived `pass_fail` from `exam_score`
6. **Exploratory Data Analysis (EDA)** — class distribution, feature histograms, correlation heatmap
7. **Feature Selection** — dropped `student_id` and `exam_score` to avoid data leakage
8. **Define Features & Target** — split into `X` and `y`
9. **Train-Test Split** — 75% train / 25% test
10. **Feature Scaling** — standardized features using `StandardScaler`
11. **Model Building** — trained a `LogisticRegression` classifier
12. **Predictions** — generated predictions on the test set
13. **Evaluation** — Accuracy, Precision, Recall, and F1-score
14. **Cross-Validation** — 5-fold cross-validation using a `Pipeline` (scaler + model) to prevent data leakage across folds
15. **Save the Model** — persisted the trained model and scaler with `joblib`

---

## 📊 Results

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | 0.78   |
| Precision           | 0.89   |
| Recall              | 0.83   |
| F1 Score            | 0.86   |
| Cross-Val Accuracy  | 0.865 (5-fold, pipeline-scaled) |

---

## 🖥️ Web App (Streamlit)

A simple, interactive **Streamlit** app (`app.py`) lets users input a student's details and get an instant Pass/Fail prediction with a confidence score.

### Run locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Python**
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualization
- **scikit-learn** — model training, scaling, evaluation, pipelines
- **joblib** — model persistence
- **Streamlit** — web app / UI deployment

---

## 📸 Screenshots

See the `ScreenShots/` folder for UI previews.

---

## 👤 Author

Junaid Ur Rehman