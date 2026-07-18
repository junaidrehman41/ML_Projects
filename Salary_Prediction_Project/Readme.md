# Salary Prediction using Simple Linear Regression
**Live Demo:** [https://junaidrehman41-ml-projects-salary-prediction-projectapp-05hpzb.streamlit.app/](https://junaidrehman41-ml-projects-salary-prediction-projectapp-05hpzb.streamlit.app/)

## Project Overview
This project builds a Machine Learning model that predicts an employee's **Salary** based on their **Years of Experience**, using the **Simple Linear Regression** algorithm.

Simple Linear Regression works by finding the best-fit straight line (`y = mx + c`) between an independent variable (input) and a dependent variable (output). In this case:
- **Independent Variable (X):** Years of Experience
- **Dependent Variable (y):** Salary

## Dataset
- **File:** `data/salary_data.csv`
- **Columns:**
  - `YearsExperience` — number of years of professional experience
  - `Salary` — corresponding salary

## Exploratory Data Analysis (EDA)
- Dataset contains 30 rows and 2 columns (`YearsExperience`, `Salary`)
- No missing values found in either column
- Years of Experience ranges from 1.2 to 10.6 years
- Salary ranges from $37,732 to $122,392

## Project Structure
```
salary-prediction-project/
│
├── data/
│   └── salary_data.csv        # Raw dataset
│
├── model/
│   └── salary_model.pkl       # Trained model (saved using joblib)
│
├── notebook.ipynb             # Main code notebook
│
└── README.md                  # Project documentation
```
## Steps Performed

1. **Data Loading** — Loaded the dataset using Pandas to work with it as a structured table (DataFrame).

2. **Data Visualization** — Plotted a scatter plot (Matplotlib) of Experience vs Salary to confirm a linear relationship existed before applying Linear Regression.

3. **Train-Test Split** — Split the dataset into 80% training data and 20% testing data using `train_test_split`, with `random_state=42` to keep the split consistent across runs. This ensures the model is evaluated on unseen data rather than the same data it was trained on.

4. **Model Training** — Trained a `LinearRegression` model (from Scikit-learn) on the training data, allowing it to learn the best-fit slope (`m`) and intercept (`c`).

5. **Prediction** — Used the trained model to predict salaries for the test set (data the model had not seen before).

6. **Model Evaluation** — Measured model performance using:
   - **R2 Score** — indicates how well the model explains the variance in the data (closer to 1 is better)
   - **MAPE (Mean Absolute Percentage Error)** — indicates the average prediction error as a percentage

7. **Model Saving** — Saved the trained model as a `.pkl` file using `joblib`, so it can be reused later (e.g., for deployment) without retraining.

## Results

| Metric | Value | Meaning |
|--------|-------|---------|
| R2 Score | 0.888 | Model explains ~88.8% of the variance in salary based on experience |
| MAPE | 10.90% | Predictions are, on average, about 10.9% off from actual salary values |

These results indicate the model performs well for this dataset, given the strong linear relationship between experience and salary.

## Dependencies
pip install pandas matplotlib scikit-learn numpy joblib

## How to Run
1. Clone or download this project folder.
2. Install the required dependencies (command above).
3. Open `notebook.ipynb` in VS Code or Jupyter and run all cells from top to bottom.

## Model File
The trained model is saved at `model/salary_model.pkl`. It can be loaded directly for making new predictions without retraining, e.g.:
```python
import joblib
model = joblib.load('model/salary_model.pkl')
```

## Author
[Junaid Ur Rehman]