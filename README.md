# Churn Classification - Bank Churn Classification with Machine Learning

### This project is an application of machine learning models to predict clients churns based on public data provided by Kaggle. The project is a demonstration of analytical skills and machine learning.

<p align="center">
  <img src="img_churn.png">
</p>


## Dataset

The dataset used in this project can be found on Kaggle: [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)


## Project Structure

- `bank_churn.ipynb`: Jupyter Notebook containing the code and analysis for the churn classification.
- `churn.csv`: Dataset used for training and testing the machine learning models.
- `README.md`: Project documentation.

## Project Overview

### Introduction

- **Objective:** Develop a classification model to predict whether a customer will cancel their bank account.
- **Dataset:** Sourced from Kaggle, containing information about bank customers.
- **Problem:** Address bank customer turnover using the "Exited" column in the dataset.
- **Business Assumptions:**
  1. Customers may cancel their bank account for various reasons.
  2. Dataset contains customer information such as age, gender, location, transaction history, etc.
  3. Customer information was collected over time to identify patterns.
  4. Customers who did not carry out transactions within 3 months were considered churn.
  5. The model must distinguish between customers likely to cancel and those who are not.

### Data Preprocessing
- **Steps:**
  1. Read the dataset.
  2. Rename columns to snake_case.
  3. Drop unnecessary columns (e.g., 'row_number').
  4. Calculate and visualize churn rate.
  5. Check data types, missing values, and duplicates.
  6. Provide descriptive statistics for numeric and categorical variables.
  7. Visualize categorical variables.
  8. Drop the 'surname' column.

### Exploratory Data Analysis (EDA)
- **Key Insights:**
  - Customers aged over 45 are more likely to leave the bank.
  - The proportion of churns is higher for females (25.1%) compared to males (16.5%).
  - Active members tend not to churn, while non-active members tend to churn.
  - Germany has the highest proportion of customers who have churned (32.4%).
  - Customers with more than two services/products have a higher churn rate.

### Feature Engineering
- **New Features:**
  - `tenure_by_age`: Tenure divided by age.
  - `age_binned`: Age divided into bins.
- Encoding:
  - One-hot encoding for categorical variables.
- Rescaling:
  - MinMaxScaler for numeric variables.

### Hypothesis Testing
- **Hypotheses:**
  1. Older age increases churn rate (True).
  2. Churn rate increases with tenure (False).
  3. Being an active member decreases churn rate (True).
  4. Clients with more money are more likely to churn (True).
  5. Clients with a credit card are less likely to churn (False).

### Train-Test Split
- **Split:** 80% training and 20% test.

### Feature Importance

- **Methods:**
  1. Decision Tree (Gini Impurity).
  2. Logistic Regression coefficients.
  3. Boruta algorithm.

- **Selected Features:**
  - Age, number of products, estimated salary, balance.

### Machine Learning Models
- **Models Evaluated:**
  - Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, SVM, Gradient Boosting, LightGBM.
- **Best Model:** LightGBM chosen for further tuning.

### Model Tuning
- **Grid Search:** Performed to find the best hyperparameters for LightGBM.

- **Evaluation:**
  - ROC AUC: 86%
  - Confusion Matrix: 85.65% correct classifications, 14.35% incorrect classifications.
  - Precision and Recall: Evaluated for both churn and non-churn classes.

### Conclusions

- The model performs well in classifying non-churn customers but struggles with churn customers.
- Considerations for improvement:
  - Data balancing techniques.
  - Further feature engineering.
  - Optimizing feature selection.
  - Exploring other models and tuning parameters.
