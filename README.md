# Interest Rate Direction Forecasting using XGBoost

## Overview

This project builds a **machine learning pipeline to forecast the
direction of U.S. Federal Funds interest rate changes** using
macroeconomic indicators from the **FRED (Federal Reserve Economic
Data)** database.

The model predicts whether the Federal Reserve is likely to:

-   **Cut** interest rates\
-   **Hold** interest rates\
-   **Hike** interest rates

The pipeline combines:

-   Macroeconomic feature engineering
-   Lag-based time series predictors
-   Permutation feature importance
-   Rolling-window forecasting
-   Multi-class classification using **XGBoost**

------------------------------------------------------------------------

# Dataset

Macroeconomic data is pulled from **FRED** using `pandas_datareader`.

### Variables Used

  Variable   Description
  ---------- -----------------------------
  FEDFUNDS   Federal Funds Rate
  PCEPILFE   Core PCE Inflation
  UNRATE     Unemployment Rate
  INDPRO     Industrial Production
  DGS10      10-Year Treasury Yield
  DGS2       2-Year Treasury Yield
  T5YIFR     5y5y Inflation Expectations
  BAA        BAA Corporate Bond Yield
  AAA        AAA Corporate Bond Yield

Time range:

1995 → 2025

The dataset is converted to **monthly frequency**.

------------------------------------------------------------------------

# Feature Engineering

Several macroeconomic indicators are constructed to reflect the economic
environment influencing monetary policy.

## Inflation Features

-   Core PCE YoY Inflation
-   Inflation Momentum

``` python
core_pce_yoy = 100 * pct_change(12)
infl_momentum = core_pce_yoy - core_pce_yoy.shift(3)
```

------------------------------------------------------------------------

## Labor Market Features

-   Natural unemployment rate estimate
-   Unemployment gap

``` python
u_star = rolling mean (60 months)
unemp_gap = unrate - u_star
```

------------------------------------------------------------------------

## Financial Market Features

-   Yield curve slope
-   Credit spread

``` python
yield_slope = gs10 - gs2
credit_spread = baa - aaa
```

------------------------------------------------------------------------

## Policy Memory Feature

Captures how long the Fed has held rates constant.

    months_since_change

------------------------------------------------------------------------

# Lag Feature Creation

Lagged versions of macro variables are created to allow the model to
capture **historical economic dynamics**.

Examples:

    inflation_lag1
    inflation_lag3
    yield_slope_lag6

These lag features are automatically detected in the pipeline.

------------------------------------------------------------------------

# Target Variable

The model predicts the **direction of future interest rate changes**.

  Class   Label   Meaning
  ------- ------- ----------------
  0       Cut     Rate decreases
  1       Hold    No change
  2       Hike    Rate increases

Target generation:

``` python
future_change = rate.shift(-horizon) - rate
```

Decision rule:

-   0.001 → **Hike**

-   \< -0.001 → **Cut**

-   Otherwise → **Hold**

------------------------------------------------------------------------

# Feature Selection

Permutation importance is used to select the **Top K predictive
features**.

Steps:

1.  Train a **Random Forest model**
2.  Compute **permutation importance**
3.  Rank features by importance
4.  Select **Top K features**

This reduces overfitting and improves interpretability.

------------------------------------------------------------------------

# Model

The final classification model uses **XGBoost**.

Advantages:

-   Handles nonlinear relationships
-   Works well with tabular macroeconomic data
-   Robust to correlated predictors
-   Strong predictive performance

------------------------------------------------------------------------

# Rolling Forecast Framework

To simulate **real-world forecasting**, the model uses a rolling window
approach.

Workflow:

1.  Train model on historical window (120 months)
2.  Predict the next observation
3.  Move the training window forward
4.  Repeat until the dataset ends

This produces **true out-of-sample predictions**.

------------------------------------------------------------------------

# Evaluation Metrics

Model performance is evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix
-   Classification Report

Class labels:

    Cut
    Hold
    Hike

Visual diagnostics include:

-   Confusion matrix heatmap
-   Actual vs predicted scatter plots

------------------------------------------------------------------------

# Project Workflow

    Data Download (FRED)
            │
    Feature Engineering
            │
    Lag Feature Creation
            │
    Permutation Feature Selection
            │
    Rolling Window Training
            │
    XGBoost Classification
            │
    Evaluation & Diagnostics

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   XGBoost
-   pandas-datareader

------------------------------------------------------------------------

# Example Usage

Run the forecasting pipeline:

``` python
TOP_K = 4

for h in [1, 3, 6]:
    detailed_diagnostics_rf(df, h, TOP_K)
```

Forecast horizons:

  Horizon   Meaning
  --------- ---------------
  1         1‑month ahead
  3         3‑month ahead
  6         6‑month ahead

------------------------------------------------------------------------

# Research Motivation

Central bank policy decisions depend on:

-   Inflation dynamics
-   Labor market conditions
-   Financial market signals

This project explores whether **machine learning models can capture
these relationships and predict policy actions**.

------------------------------------------------------------------------

# Future Improvements

Possible extensions:

-   Add more macroeconomic indicators
-   Use SHAP values for explainability
-   Compare models:
    -   Random Forest
    -   XGBoost
    -   LightGBM
    -   Logistic Regression
-   Hyperparameter tuning
-   Regime-switching models

------------------------------------------------------------------------

# Author

**Nitin Yadav**

Machine Learning • Data Science • Macroeconomic Forecasting
