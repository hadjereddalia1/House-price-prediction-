# 🏠 House Price Prediction using Regression

This project is a machine learning solution to predict house sale prices based on various features using regression models. It includes data preprocessing, feature selection, model training, evaluation, and deployment as a simple web app.

---

## 📊 Problem Statement

The goal is to build a predictive model that estimates house prices from a dataset of housing features. The dataset includes numerical and categorical variables such as square footage, number of rooms, location, and construction year.

---

## 📁 Dataset

- Source:kaggle 
- Target Variable: `SalePrice`
- Features: Mixture of numerical and categorical attributes

---

## 🧠 Solution Approach

1. **Data Cleaning & Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature engineering (e.g., lags, log transforms)

2. **Feature Selection**
   - Pearson correlation for numerical features vs SalePrice
   - ANOVA test for categorical features vs SalePrice
   - Only significant variables (p < 0.05) were retained

3. **Model Training**
   - Used regression models (e.g., Linear Regression, Random Forest, etc.)
   - Evaluated using RMSE

4. **Model Deployment**
   - A simple web application using **Streamlit**
   - Users input house features and receive a predicted sale price


