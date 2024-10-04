# Diabetes Onset Prediction Using Pima Indian Diabetes Dataset

## Project Overview
This project aims to predict the likelihood of diabetes onset among **Pima Indian women** using a variety of machine learning algorithms. The dataset includes various medical and demographic features like glucose concentration, BMI, age, and more. 

## Objective
The goal is to develop and compare different machine learning models to classify whether a patient is likely to develop diabetes based on their clinical and demographic data.

## Techniques Used
- **Data Normalization**: Ensures all features are on a similar scale, important for many machine learning algorithms.
- **Feature Scaling**: Applied using **MinMaxScaler** to scale numerical features between 0 and 1.
- **Model Development**: Implemented three models to compare performance:
  - **Support Vector Machine (SVM)**
  - **Logistic Regression**
  - **XGBoost**

## Dataset
The dataset used is the [Pima Indian Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database), containing 768 instances and 8 clinical features, which is widely used for binary classification tasks in healthcare.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/diabetes-onset-prediction.git
    ```

2. **Install the required dependencies** using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and place it in the `data/` directory.

4. **Run training scripts** to train each model:
    - **For Logistic Regression**:
      ```bash
      python src/logistic_regression.py
      ```
    - **For SVM**:
      ```bash
      python src/svm.py
      ```
    - **For XGBoost**:
      ```bash
      python src/xgboost.py
      ```

5. **Run the prediction script**:
    ```bash
    python src/predict.py
    ```

## Dependencies
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
