"# forecasting_econometeric_indicators_using_machine_learning_models" 
# Interest Rate Forecasting using Rolling Random Forest and FRED Macroeconomic Data

## Overview

This project builds a **machine learning framework to forecast the direction of U.S. Federal Reserve interest rate changes** using macroeconomic and financial indicators.

The model uses:

* **Macroeconomic data from the FRED database**
* **Feature engineering from economic indicators**
* **Permutation-based feature selection**
* **Rolling window Random Forest classification**

The goal is to classify future interest rate movements into three categories:

| Class | Label | Meaning                               |
| ----- | ----- | ------------------------------------- |
| 0     | Cut   | Interest rate expected to decrease    |
| 1     | Hold  | Interest rate expected to stay stable |
| 2     | Hike  | Interest rate expected to increase    |

The model evaluates predictions across **multiple forecasting horizons**.

---

# Data Source

The dataset is obtained from the **FRED (Federal Reserve Economic Data) database** using `pandas_datareader`.

Macroeconomic indicators used:

| Variable | Description                 |
| -------- | --------------------------- |
| FEDFUNDS | Federal Funds Rate          |
| PCEPILFE | Core PCE Inflation          |
| UNRATE   | Unemployment Rate           |
| INDPRO   | Industrial Production       |
| DGS10    | 10-Year Treasury Yield      |
| DGS2     | 2-Year Treasury Yield       |
| T5YIFR   | 5Y5Y Inflation Expectations |
| BAA      | BAA Corporate Bond Yield    |
| AAA      | AAA Corporate Bond Yield    |

Time range:

1995 – 2025

---

# Feature Engineering

Several macroeconomic indicators are transformed into predictive features.

### Inflation Indicators

* Year-over-year core inflation
* Inflation momentum

Example:

```
core_pce_yoy = 100 * core_pce.pct_change(12)
infl_momentum = core_pce_yoy - core_pce_yoy.shift(3)
```

---

### Labor Market Indicators

* Natural unemployment rate estimate
* Unemployment gap

```
u_star = rolling_mean(unemployment, 60 months)
unemp_gap = unemployment - u_star
```

---

### Financial Conditions

* Yield curve slope
* Credit spread

```
yield_slope = 10yr_yield - 2yr_yield
credit_spread = BAA - AAA
```

---

### Lag Features

The model automatically generates lagged predictors such as:

```
rate_lag1
rate_lag2
infl_momentum_lag3
yield_slope_lag2
```

These capture **temporal dynamics of macroeconomic variables**.

---

# Target Variable

The model predicts **future interest rate direction**.

Future change is computed as:

```
future_change = rate(t + horizon) - rate(t)
```

Classification rule:

| Condition               | Class |
| ----------------------- | ----- |
| future_change < -0.001  | Cut   |
| -0.001 ≤ change ≤ 0.001 | Hold  |
| future_change > 0.001   | Hike  |

---

# Feature Selection

The project uses **Permutation Feature Importance** to identify the most informative predictors.

Procedure:

1. Train Random Forest on the first training window
2. Randomly shuffle feature values
3. Measure drop in model performance
4. Rank features by importance
5. Select **Top-K most important predictors**

Example output:

```
Top Features by Permutation Importance
yield_slope_lag2
infl_momentum_lag3
unemp_gap_lag1
rate_lag1
```

---

# Rolling Forecast Framework

A **rolling window approach** simulates real-time forecasting.

Training window:

```
120 months
```

Forecast process:

```
Train on past data
↓
Predict future horizon
↓
Move window forward
↓
Repeat
```

This avoids **look-ahead bias** and mimics real-world forecasting conditions.

---

# Model

The classification model used is **Random Forest**.

Parameters:

```
n_estimators = 200
max_depth = 6
min_samples_leaf = 5
random_state = 42
```

These settings balance:

* predictive accuracy
* generalization
* computational efficiency

---

# Forecast Horizons

The model evaluates forecasts for multiple time horizons.

| Horizon  | Interpretation                |
| -------- | ----------------------------- |
| 1 Month  | Short-term policy prediction  |
| 3 Months | Medium-term policy prediction |
| 6 Months | Longer-term policy trend      |

---

# Evaluation Metrics

The model reports several classification metrics:

* Accuracy
* Precision (Macro Average)
* Recall (Macro Average)
* F1 Score
* Confusion Matrix
* Classification Report

Example output:

```
precision    recall  f1-score   support

Cut       0.33      0.40      0.36        25
Hold      0.44      0.21      0.29        19
Hike      0.75      0.80      0.77        88
```

---

# Visualizations

The notebook produces diagnostic plots:

### Confusion Matrix

Shows model prediction accuracy across classes.

### Actual vs Predicted Scatter Plot

Displays the alignment between predicted and actual policy decisions over time.

---

# Project Workflow

```
Download FRED Data
        ↓
Feature Engineering
        ↓
Lag Feature Creation
        ↓
Target Construction
        ↓
Permutation Feature Selection
        ↓
Rolling Random Forest Model
        ↓
Evaluation Metrics
        ↓
Visualization
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/fed-rate-forecasting.git
cd fed-rate-forecasting
```

Install required libraries:

```
pip install pandas numpy matplotlib scikit-learn pandas_datareader
```

---

# Running the Model

Run the forecasting pipeline:

```
TOP_K = 4

for h in [1,3,6]:
    detailed_diagnostics_rf(df, h, TOP_K)
```

This will output:

* selected features
* classification metrics
* confusion matrix
* prediction plots

---

# Future Improvements

Potential extensions include:

* XGBoost / LightGBM models
* Hyperparameter optimization
* Additional macroeconomic indicators
* Time-series cross-validation
* Probabilistic policy forecasts

---

# Author

Machine Learning Project focused on **Macroeconomic Forecasting and Monetary Policy Prediction**.

This repository demonstrates how machine learning can be applied to **financial time-series and macroeconomic policy analysis**.

