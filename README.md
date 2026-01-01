# Student Math Score Predictor

A machine learning web application that predicts a student's math score based on their demographic information and other test scores.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How Each File Works](#how-each-file-works)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Models Trained](#models-trained)
7. [Installation](#installation)
8. [How to Run](#how-to-run)
9. [API Endpoints](#api-endpoints)
10. [Results](#results)

---

## Project Overview

This project predicts a student's math score using machine learning. The user inputs information like gender, parental education level, lunch type, test preparation status, and their reading/writing scores. The trained model then predicts their likely math score.

**Problem Type:** Regression (predicting a continuous value between 0-100)

**Best Model:** Lasso Regression with 88.21% R-squared score (after hyperparameter tuning)

---

## Dataset

**Source:** Kaggle - Students Performance in Exams

**Size:** 1000 students, 8 columns

**Features:**

| Column | Type | Description |
|--------|------|-------------|
| gender | Categorical | Male or Female |
| race_ethnicity | Categorical | Group A, B, C, D, or E |
| parental_level_of_education | Categorical | Highest education level of parents |
| lunch | Categorical | Standard or Free/Reduced |
| test_preparation_course | Categorical | Completed or None |
| reading_score | Numerical | Score out of 100 |
| writing_score | Numerical | Score out of 100 |
| math_score | Numerical | Target variable (what we predict) |

**Key Insights from EDA:**
- Students with standard lunch score higher than those with free/reduced lunch
- Completing test preparation course improves scores
- Reading, writing, and math scores are linearly correlated
- Females have higher pass rates and more top scorers

---

## Project Structure

```
student-performance-predictor/
|
|-- artifacts/                       [Generated after training]
|   |-- model.pkl                    Trained model saved as pickle file
|   |-- preprocessor.pkl             Data transformer saved as pickle file
|   |-- train.csv                    Training dataset (80%)
|   |-- test.csv                     Testing dataset (20%)
|
|-- src/
|   |-- components/
|   |   |-- data_ingestion.py        Loads data and splits into train/test
|   |   |-- data_transformation.py   Converts raw data into model-ready format
|   |   |-- model_trainer.py         Trains multiple models and selects best one
|   |
|   |-- pipeline/
|   |   |-- predict_pipeline.py      Handles predictions for new data
|   |
|   |-- exception.py                 Custom exception handling
|   |-- logger.py                    Logging configuration
|   |-- utils.py                     Helper functions (save/load objects)
|
|-- templates/
|   |-- index.html                   Landing page
|   |-- home.html                    Prediction form and results page
|
|-- notebook/
|   |-- 1_EDA_STUDENT_PERFORMANCE.ipynb      Exploratory Data Analysis
|   |-- 2_MODEL_TRAINING.ipynb               Model experimentation
|
|-- app.py                           Flask web application
|-- requirements.txt                 Python dependencies
|-- setup.py                         Package configuration
|-- README.md                        Project documentation
```

---

## How Each File Works

### data_ingestion.py

This file is responsible for loading and splitting the data.

**What it does:**
1. Reads the raw CSV file containing student data
2. Splits data into training set (80%) and testing set (20%)
3. Saves the splits as train.csv and test.csv in the artifacts folder

**Why we split:**
We train the model on 80% of data and test it on the remaining 20% to see how well it performs on unseen data. This prevents overfitting.

---

### data_transformation.py

This file converts raw data into a format the model can understand.

**What it does:**

1. Handles categorical columns (text to numbers):
   - Uses OneHotEncoder to convert categories into binary columns
   - Example: gender "female" becomes [1, 0], "male" becomes [0, 1]
   - Converts 5 categorical columns into 17 binary columns

2. Handles numerical columns:
   - Uses StandardScaler to normalize reading_score and writing_score
   - Converts values to have mean=0 and standard deviation=1
   - Example: score of 72 might become 0.15 after scaling

3. Combines both transformers using ColumnTransformer

4. Saves the transformer as preprocessor.pkl so we can apply the same transformation to new data during prediction

**Final output:** 19 features (17 from one-hot encoding + 2 scaled numerical)

---

### model_trainer.py

This file trains multiple models and picks the best one.

**What it does:**
1. Takes the transformed training data
2. Trains 9 different regression models
3. Evaluates each model on the test set using R-squared score
4. Selects the model with highest test R-squared
5. Saves the best model as model.pkl

**Why multiple models:**
Different algorithms work better for different types of data. By comparing 9 models, we find which one generalizes best for this specific problem.

---

### predict_pipeline.py

This file handles predictions for new data.

**What it does:**
1. Loads the saved model.pkl and preprocessor.pkl
2. Takes new input data from the user
3. Transforms the input using the preprocessor (same transformation as training)
4. Passes transformed data to the model
5. Returns the predicted math score

---

### utils.py

Helper functions used across the project.

**Functions:**
- save_object(): Saves Python objects (model, preprocessor) as .pkl files
- load_object(): Loads .pkl files back into Python objects
- evaluate_models(): Trains multiple models and returns their scores

---

### exception.py

Custom exception handling for better error messages.

**What it does:**
- Catches errors and shows which file and line number caused the problem
- Makes debugging easier by providing detailed error information

---

### logger.py

Logging configuration for tracking what happens during execution.

**What it does:**
- Creates log files with timestamps
- Records important events (data loaded, model trained, etc.)
- Helps debug issues by showing the sequence of operations

---

### app.py

The Flask web application that ties everything together.

**What it does:**
1. Creates a web server on your computer
2. Defines URL routes (endpoints)
3. Renders HTML templates
4. Handles form submissions
5. Calls the prediction pipeline and returns results

---

## Machine Learning Pipeline

The complete flow from raw data to prediction:

```
TRAINING PHASE:

Raw CSV --> data_ingestion.py --> train.csv, test.csv
                                       |
                                       v
train.csv --> data_transformation.py --> Transformed data + preprocessor.pkl
                                              |
                                              v
Transformed data --> model_trainer.py --> model.pkl (best model saved)


PREDICTION PHASE:

User Input --> predict_pipeline.py --> Load preprocessor.pkl --> Transform input
                                              |
                                              v
                                       Load model.pkl --> Predict --> Math Score
```

---

## Models Trained

Ten regression models were trained and compared. After hyperparameter tuning using GridSearchCV/RandomizedSearchCV, here are the final results:

| Rank | Model | Test R-squared | Notes |
|------|-------|----------------|-------|
| 1 | Lasso Regression | 0.8821 | Best after tuning, L1 regularization |
| 2 | Ridge Regression | 0.8804 | L2 regularization, close second |
| 3 | CatBoosting Regressor | 0.8742 | Gradient boosting for categorical data |
| 4 | Linear Regression | 0.8738 | Simple baseline, no regularization |
| 5 | Gradient Boosting | 0.8722 | Ensemble of weak learners |
| 6 | XGBRegressor | 0.8667 | Extreme gradient boosting |
| 7 | Random Forest | 0.8524 | Bagging ensemble of decision trees |
| 8 | AdaBoost Regressor | 0.8515 | Adaptive boosting |
| 9 | Decision Tree | 0.8241 | Single tree, prone to overfitting |
| 10 | K-Neighbors Regressor | 0.5786 | Distance-based, poor for one-hot encoded data |

**Why Lasso Won (after tuning):**
- L1 regularization adds penalty using absolute values of weights
- Can zero out irrelevant features (feature selection built-in)
- With proper alpha parameter tuning, it found the optimal regularization strength
- Generalized better than Ridge on this specific dataset

**Why K-Neighbors Failed:**
- Only 57.86% R-squared, much worse than other models
- KNN uses Euclidean distance to find similar data points
- One-hot encoded categorical data creates sparse high-dimensional space
- Distance metrics become less meaningful in high dimensions (curse of dimensionality)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
# Mac/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

1. Make sure virtual environment is activated

2. Run the Flask application:
```bash
# Mac/Linux
./venv/bin/python app.py

# Or if venv is activated
python app.py
```

3. Open your browser and go to:
```
http://127.0.0.1:5000/
```

**What is 127.0.0.1:5000?**
- 127.0.0.1 = localhost = your own computer
- 5000 = port number (like a door number for the application)
- Together they mean "connect to the Flask app running on my computer"

---

## API Endpoints

| URL | Method | Description |
|-----|--------|-------------|
| / | GET | Landing page with project introduction |
| /predictdata | GET | Shows the prediction form |
| /predictdata | POST | Submits form data and returns predicted score |

**How the form submission works:**
1. User fills out the form at /predictdata
2. Clicks "Predict Math Score" button
3. Browser sends POST request with form data
4. Flask receives data, calls predict_pipeline.py
5. Model makes prediction
6. Flask renders home.html with the result

---

## Results

**Final Model:** Lasso Regression (after hyperparameter tuning)

**Performance Metrics:**
- R-squared Score: 0.8821 (88.21% of variance explained)
- Mean Absolute Error: approximately 4-5 points

**What R-squared means:**
- R-squared of 0.88 means the model explains 88% of the variation in math scores
- The remaining 12% is due to factors not captured in our features
- For educational prediction, this is considered a good score

**Interpretation:**
- On average, predictions are off by about 4-5 points
- A predicted score of 75 likely means actual score is between 70-80

**Why Hyperparameter Tuning Mattered:**
- Before tuning: Lasso scored 0.8253 (rank 6)
- After tuning: Lasso scored 0.8821 (rank 1)
- Finding the right alpha (regularization strength) made Lasso jump from 6th to 1st place

---

## Technologies Used

- Python 3.10
- Scikit-learn (machine learning)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Flask (web framework)
- HTML/CSS (frontend)
- Pickle (model serialization)

---

## Future Improvements

1. Add more features like study hours and attendance
2. Try neural networks with proper regularization
3. Deploy to cloud using AWS, Heroku, or Render
4. Add model monitoring for data drift detection
5. Create API documentation with Swagger
6. Add unit tests for each component
7. Containerize with Docker for easier deployment

---

## Author

Vedant

---

## License

This project is for educational purposes.