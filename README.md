# Medical Insurance Cost Predictor

An end-to-end Machine Learning project that predicts annual medical insurance costs based on personal and health information.

**Best Model:** XGBoost Regressor  
**Accuracy:** 88.39% R-squared  
**Dataset:** 1,338 records  

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How Each File Works](#how-each-file-works)
5. [ML Pipeline](#ml-pipeline)
6. [Model Performance](#model-performance)
7. [Why XGBoost Won](#why-xgboost-won)
8. [Feature Importance](#feature-importance)
9. [Installation](#installation)
10. [How to Run](#how-to-run)
11. [API Endpoints](#api-endpoints)
12. [Technologies Used](#technologies-used)

---

## Overview

**Problem:** Predict how much a person will pay for annual medical insurance based on their demographics and health information.

**Approach:**
- Trained 10 different regression models
- Used GridSearchCV for hyperparameter tuning with 5-fold cross-validation
- Selected XGBoost as the best performer
- Deployed as a Flask web application

**Input Features:** Age, Sex, BMI, Number of Children, Smoker Status, Region

**Output:** Predicted annual insurance cost in US dollars

---

## Dataset

Source: Kaggle - Medical Cost Personal Dataset

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| age | Numerical | Age of person | 18 - 64 |
| sex | Categorical | Gender | male, female |
| bmi | Numerical | Body Mass Index | 15.96 - 53.13 |
| children | Numerical | Number of dependents | 0 - 5 |
| smoker | Categorical | Smoking status | yes, no |
| region | Categorical | US geographic region | northeast, northwest, southeast, southwest |
| charges | Numerical | TARGET - Annual cost | $1,121 - $63,770 |

Dataset Statistics:
- Total records: 1,338
- Training set: 1,070 (80%)
- Test set: 268 (20%)
- Missing values: 0

---

## Project Structure

```
medical-insurance-predictor/
│
├── notebook/
│   └── data/
│       └── insurance.csv
│
├── src/
│   ├── components/
│   │   ├── artifacts/
│   │   │   ├── model.pkl
│   │   │   ├── preprocessor.pkl
│   │   │   ├── train.csv
│   │   │   └── test.csv
│   │   │
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── hyperparameter_tuning.py
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   ├── home.html
│   └── technical.html
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## How Each File Works

### src/components/data_ingestion.py

Reads the insurance.csv file and splits it into training (80%) and test (20%) sets. Saves the split data to the artifacts folder.

```python
# Key operation
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
```

### src/components/data_transformation.py

Creates the preprocessing pipeline using ColumnTransformer:

**Numerical features (age, bmi, children):**
- SimpleImputer with median strategy
- StandardScaler for normalization

**Categorical features (sex, smoker, region):**
- SimpleImputer with most frequent strategy
- OneHotEncoder to convert text to numbers
- StandardScaler for normalization

```python
preprocessor = ColumnTransformer([
    ("num_pipeline", num_pipeline, ["age", "bmi", "children"]),
    ("cat_pipeline", cat_pipeline, ["sex", "smoker", "region"])
])
```

### src/components/hyperparameter_tuning.py

Defines parameter grids for GridSearchCV:

```python
# Example: XGBoost parameters
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}
```

Uses 5-fold cross-validation with R-squared scoring to find optimal parameters.

### src/components/model_trainer.py

Trains 10 different regression models:
1. Linear Regression
2. Ridge Regression (L2 regularization)
3. Lasso Regression (L1 regularization)
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. CatBoost
9. AdaBoost
10. K-Neighbors Regressor

Evaluates each model on the test set and selects the one with highest R-squared score. Saves the best model as model.pkl.

### src/pipeline/predict_pipeline.py

Loads the saved model.pkl and preprocessor.pkl files. Takes user input, transforms it using the preprocessor, and returns the prediction.

```python
class CustomData:
    def __init__(self, age, sex, bmi, children, smoker, region):
        # Stores user input
        
    def get_data_as_data_frame(self):
        # Converts input to DataFrame for prediction
```

### src/utils.py

Helper functions:
- save_object(): Saves Python objects as .pkl files using pickle
- load_object(): Loads .pkl files
- evaluate_models(): Calculates R-squared score for each model

### src/exception.py

Custom exception class that captures file name, line number, and error message for debugging.

### src/logger.py

Creates timestamped log files in the logs/ folder for tracking pipeline execution.

### app.py

Flask web application with three routes:
- `/` : Landing page
- `/predictdata` : Prediction form (GET shows form, POST returns prediction)
- `/technical` : Technical details page

---

## ML Pipeline

### Training Pipeline

```
insurance.csv
     |
     v
[Data Ingestion]
     |
     ├── train.csv (80%)
     └── test.csv (20%)
            |
            v
[Data Transformation]
     |
     ├── Numerical: SimpleImputer (median) -> StandardScaler
     └── Categorical: SimpleImputer (mode) -> OneHotEncoder -> StandardScaler
            |
            v
[Hyperparameter Tuning]
     |
     └── GridSearchCV with 5-fold CV for each model
            |
            v
[Model Training]
     |
     └── Train 10 models, evaluate on test set
            |
            v
[Model Selection]
     |
     └── Select best model (XGBoost, R² = 0.8839)
            |
            v
[Save Artifacts]
     |
     ├── model.pkl
     └── preprocessor.pkl
```

### Prediction Pipeline

```
User Input (age, sex, bmi, children, smoker, region)
     |
     v
[Load preprocessor.pkl]
     |
     └── Transform input using fitted preprocessor
            |
            v
[Load model.pkl]
     |
     └── Make prediction
            |
            v
Output: Predicted Insurance Cost ($)
```

---

## Model Performance

Results after hyperparameter tuning (sorted by R-squared):

| Rank | Model | R² Score | Model Type |
|------|-------|----------|------------|
| 1 | XGBRegressor | 0.8839 | Gradient Boosting |
| 2 | CatBoosting Regressor | 0.8805 | Gradient Boosting |
| 3 | Gradient Boosting | 0.8802 | Ensemble |
| 4 | Random Forest | 0.8774 | Ensemble |
| 5 | Decision Tree | 0.8683 | Tree-Based |
| 6 | AdaBoost Regressor | 0.8553 | Ensemble |
| 7 | K-Neighbors Regressor | 0.8130 | Instance-Based |
| 8 | Ridge Regression | 0.7833 | Linear (L2) |
| 9 | Linear Regression | 0.7832 | Linear |
| 10 | Lasso Regression | 0.7812 | Linear (L1) |

Key Observation: Tree-based models outperform linear models by approximately 10% R-squared.

---

## Why XGBoost Won

### Why Linear Models (Lasso, Ridge) Performed Poorly

Linear models assume: `cost = w1*age + w2*bmi + w3*smoker + ...`

This fails on this dataset for three reasons:

**1. Non-Linear Age Effect**

The relationship between age and cost is not linear. Cost accelerates as age increases:

```
Age 20 -> $2,500
Age 40 -> $7,000
Age 60 -> $15,000
```

Linear models can only draw straight lines. XGBoost can capture curves.

**2. The Smoker Effect (Sharp Jump)**

Smokers pay 3-4x more than non-smokers. This is a multiplicative effect, not additive.

```
Non-smoker, age 30: $4,000
Smoker, age 30: $35,000
```

Linear model tries to add a fixed amount for smoking. But the real relationship is:

```
# What linear models do (wrong):
cost = base + 15000 * smoker

# What actually happens (XGBoost captures this):
if smoker == yes:
    cost = 20000 + 500*age + 800*bmi
else:
    cost = 2000 + 200*age + 300*bmi
```

**3. Feature Interactions**

The smoking penalty depends on age. Older smokers pay much more than younger smokers:

```
Age 25 + Smoker: +$17,000 penalty
Age 55 + Smoker: +$33,000 penalty
```

Linear models apply the same penalty regardless of age. XGBoost detects these interactions automatically by creating different decision paths.

### Why XGBoost Excels

1. Captures non-linear relationships through tree splits
2. Handles sharp jumps by creating separate branches for smokers vs non-smokers
3. Detects feature interactions automatically
4. Built-in L1 and L2 regularization prevents overfitting
5. Combines hundreds of weak trees into one strong predictor

---

## Feature Importance

Based on XGBoost feature importance (approximate):

```
smoker      ████████████████████████████████████████  ~65%
age         ████████████                              ~15%
bmi         █████████                                 ~12%
children    ███                                       ~4%
region      ██                                        ~2%
sex         ██                                        ~2%
```

Smoking status alone accounts for approximately 65% of the prediction power.

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/medical-insurance-predictor.git
cd medical-insurance-predictor
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Train the Model

This will run data ingestion, transformation, and model training. It generates model.pkl and preprocessor.pkl in the artifacts folder.

```bash
python -m src.components.data_ingestion
```

Expected output:
```
============================================================
                  MODEL PERFORMANCE REPORT                  
============================================================
Model Name                            R² Score
------------------------------------------------------------
XGBRegressor                            0.8839
CatBoosting Regressor                   0.8805
Gradient Boosting                       0.8802
...
============================================================
                    BEST MODEL SELECTED                     
============================================================
Model: XGBRegressor
Final R² Score: 0.8839
============================================================
```

### Step 2: Run the Flask App

```bash
python app.py
```

### Step 3: Open in Browser

```
http://127.0.0.1:5001/
```

The address 127.0.0.1 means localhost (your own computer). Port 5001 is where the Flask server is running.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Landing page with project overview |
| /predictdata | GET | Shows the prediction form |
| /predictdata | POST | Submits form and returns predicted cost |
| /technical | GET | Technical details about the ML pipeline |

---

## Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3.10 |
| ML Libraries | scikit-learn, XGBoost, CatBoost |
| Data Processing | pandas, numpy |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Model Serialization | pickle |

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
catboost
flask
dill
```

---

## Example Predictions

| Age | Sex | BMI | Children | Smoker | Region | Predicted Cost |
|-----|-----|-----|----------|--------|--------|----------------|
| 25 | male | 22.5 | 0 | no | southwest | ~$3,500 |
| 35 | female | 28.0 | 2 | no | northeast | ~$7,200 |
| 45 | male | 30.0 | 1 | yes | southeast | ~$38,000 |
| 55 | female | 25.0 | 0 | no | northwest | ~$12,500 |

---

## Summary

This project demonstrates a complete ML workflow:

1. Data ingestion and train/test split
2. Preprocessing with ColumnTransformer (OneHotEncoder + StandardScaler)
3. Hyperparameter tuning with GridSearchCV
4. Model comparison (10 models)
5. Best model selection (XGBoost with 88.39% R²)
6. Flask deployment with web interface

The key insight is that tree-based models significantly outperform linear models on this dataset because insurance pricing has non-linear relationships, sharp jumps (smoker effect), and feature interactions that linear models cannot capture.