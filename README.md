# Predicting Depression Among University Students

## Overview

This project aims to predict depression among university students using survey data collected from students at the International Islamic University Malaysia (IIUM). The dataset, sourced from Kaggle, includes demographic details, academic performance indicators, and self-reported mental health conditions such as depression, anxiety, and panic attacks. The goal is to leverage machine learning to identify students at risk of depression for early intervention and support.

## Dataset

- **Source**: Kaggle - Student Mental Health
- **Features**:
  - Age
  - Gender
  - Course of study
  - Year of study
  - CGPA (Cumulative Grade Point Average)
  - Marital status
  - Mental health indicators: Depression (Yes/No), Anxiety (Yes/No), Panic attacks (Yes/No)
  - Treatment-seeking behavior
- **Size**: 101 student responses
- **Collection Method**: Google Forms survey conducted by the dataset author.

## Methodology

1. **Data Preprocessing**:

   - Loaded dataset from `Student Mental health.csv`.
   - Handled missing values (if any) and cleaned data.
   - Encoded categorical variables (e.g., Gender, Course, Year of Study, Marital Status) using techniques like One-Hot Encoding or Label Encoding.
   - Scaled numerical features (e.g., Age) using MinMaxScaler for model compatibility.

2. **Exploratory Data Analysis (EDA)**:

   - Visualized distributions of features (e.g., histograms, bar plots) using Matplotlib, Seaborn, and Plotly.
   - Analyzed relationships between features and depression using correlation matrices and other statistical methods.
   - Key findings:
     - Females reported higher rates of depression, anxiety, and panic attacks.
     - Married students showed increased likelihood of depression and anxiety.
     - Mid-range CGPA (3.00 - 3.49) students were more prone to depression.
     - First-year students exhibited higher depression levels.

3. **Feature Engineering**:

   - Selected relevant features based on EDA insights.
   - Potentially transformed features (e.g., categorizing CGPA into ranges) to capture non-linear relationships.

4. **Model Selection and Training**:

   - Split data into training and testing sets.
   - Trained multiple models:
     - Logistic Regression
     - Random Forest
     - Naive Bayes
     - Support Vector Machine (SVM)
     - Hyperparameter-tuned SVM (Best SVM)

5. **Model Evaluation**:

   - Metrics used: Accuracy, Precision, Recall, F1-score, AUC-ROC.
   - Visualizations included confusion matrices and ROC curves.

6. **Hyperparameter Tuning**:

   - Optimized SVM model parameters to enhance performance.

## Results

- **Model Performance**:
  - **Naive Bayes**: AUC = 0.88, F1-score = 0.77
  - **Logistic Regression**: AUC = 0.86, F1-score = 0.82 (Best overall model)
  - **Best SVM**: AUC = 0.84, F1-score = 0.79
  - **Random Forest**: AUC = Competitive, F1-score = 0.82
  - **SVM (untuned)**: AUC = Competitive, F1-score = 0.77
- **Best Model**: Logistic Regression, due to its balanced high AUC (0.86) and top F1-score (0.82).

## Insights

- **Gender**: Females are more likely to report mental health issues, including depression.
- **Marital Status**: Married students exhibit higher depression and anxiety rates.
- **CGPA**: Students with CGPA 3.00–3.49 show elevated depression risk.
- **Year of Study**: First-year students are particularly vulnerable to depression.
- These findings suggest that demographic and academic factors play a significant role in student mental health.

## Requirements

- **Python Version**: 3.x
- **Libraries**:
  - NumPy
  - Pandas
  - Matplotlib
  - Plotly
  - Scikit-learn
  - Seaborn
