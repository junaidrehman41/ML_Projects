# ML Projects

A collection of end-to-end machine learning projects — from raw data to a trained model to a deployed, interactive web application. Each project follows the same workflow: data cleaning, exploratory data analysis, model training, evaluation, and deployment with Streamlit.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

---

## 📂 Projects

### 1. Salary Prediction
Predicts an employee's expected salary based on years of experience.

- **Type:** Regression
- **Algorithm:** Simple Linear Regression
- **Performance:** R² Score 0.888 | MAPE 10.90%
- **Folder:** [`Salary Prediction Project`](./Salary%20Prediction%20Project)
- **Live Demo:** [Try it here](https://junaidrehman41-ml-projects-salary-prediction-projectapp-05hpzb.streamlit.app/)

---

### 2. Student Exam Score Prediction
Predicts whether a student will pass or fail based on study habits and academic history.

- **Type:** Classification
- **Algorithm:** Logistic Regression
- **Performance:** Accuracy 0.78 | Precision 0.89 | Recall 0.83 | F1-Score 0.86
- **Folder:** [`Student_Exam_Scores_Prediction`](./Student_Exam_Scores_Prediction)
- **Live Demo:** [Try it here](https://junaidrehman41-ml-proj-student-exam-scores-predictionapp-gh1ytt.streamlit.app/)

---

### 3. Titanic Survival Prediction
Predicts whether a passenger would have survived the Titanic disaster based on passenger attributes.

- **Type:** Classification
- **Algorithm:** Decision Tree Classifier
- **Performance:** Accuracy 0.77 | Precision 0.73 | Recall 0.70 | F1-Score 0.71
- **Folder:** [`Titanic_Survival_Prediction`](./Titanic_Survival_Prediction)
- **Live Demo:** [Try it here](https://junaidrehman41-ml-project-titanic-survival-predictionapp-e3xazk.streamlit.app/)

---

## 🖥️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Deployment | Streamlit |

---

## 📁 Repository Structure

```
ML_Projects/
├── Salary Prediction Project/
│   ├── Data/
│   ├── Notebooks/
│   ├── model/
│   ├── app.py
│   └── requirements.txt
├── Student_Exam_Scores_Prediction/
│   ├── Data/
│   ├── Notebooks/
│   ├── Model/
│   ├── app.py
│   └── requirements.txt
├── Titanic_Survival_Prediction/
│   ├── Data/
│   ├── Notebooks/
│   ├── Model/
│   ├── images/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

Each project folder is self-contained with its own dataset, notebook, trained model, Streamlit app, and dependencies.

---

## ⚙️ Running a Project Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/junaidrehman41/ML_Projects.git
   cd ML_Projects
   ```

2. Navigate into the project you want to run:
   ```bash
   cd "Salary Prediction Project"
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author

**Junaid Ur Rehman**
GitHub: [github.com/junaidrehman41](https://github.com/junaidrehman41)
Email: junaidrehman41@gmail.com