# Walmart Sales Forecasting

An end-to-end data science project: weekly sales forecasting for 45 Walmart stores and 81 departments. The project covers the full pipeline, from SQL-based data loading and merging to feature engineering and model comparison.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tools Used](#tools-used)
- [Methodology](#methodology)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

## Problem Statement

This project aims to forecast weekly sales at the store-department level for Walmart, using historical sales, economic indicators (CPI, unemployment rate, fuel price), promotional (MarkDown) data, and holiday periods. Accurately predicting sales fluctuations around holiday periods is of particular business importance.

## Dataset

The dataset was sourced from [Kaggle - Walmart Sales Forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) and consists of 4 files:

| File | Content |
|---|---|
| `train.csv` | Weekly sales data from Feb 2010 – Oct 2012 (421,570 rows) |
| `test.csv` | Period to be forecasted, Nov 2012 – Jul 2013 (115,064 rows) |
| `stores.csv` | Store type (A/B/C) and size |
| `features.csv` | Temperature, CPI, unemployment, fuel price, promotions (MarkDown1-5) |

## Tools Used

- **Python** (pandas, numpy, matplotlib)
- **SQLite** — for data merging and analytical queries
- **scikit-learn** — Linear Regression, Random Forest, hyperparameter tuning
- **XGBoost** — gradient boosting model

## Methodology

### 1. Data Loading & Merging
The 4 CSV files were loaded into a SQLite database and merged into a single table via JOINs on `Store` and `Store+Date`.

### 2. Exploratory Analysis & Data Cleaning
- Missing values in the MarkDown columns (~50-65%) were filled with `0` — missing simply means no promotion occurred that week.
- Missing CPI/Unemployment values were investigated and confirmed to fall exclusively within the last 13 weeks of the test period (May–Jul 2013).
- Negative sales values (1,285 rows, ~0.3%) were analyzed — found to be concentrated in specific departments (notably Dept 47) rather than randomly distributed, and were clipped to `0`.

### 3. Feature Engineering
- Extracted `Year`, `Month`, `WeekOfYear`, and `Session` (season) from the date.
- Added `lag_1`, `lag_4`, and `lag_52` — lag features carrying past sales information.
- Created `IsBeforeHoliday` / `IsAfterHoliday` flags and validated their effect: average sales are **13-14% higher** in the week before a holiday.
- One-hot encoded the `Type` and `Session` columns.

### 4. Train/Validation Split
The data was split **chronologically**, not randomly — a critical step to avoid data leakage in time-series data.

### 5. Modeling & Hyperparameter Tuning
Three models were compared: Linear Regression (baseline), Random Forest, and XGBoost. Both tree-based models were tuned via `RandomizedSearchCV`, using **`TimeSeriesSplit`** instead of standard K-Fold for cross-validation — a choice that led to a different (and better-performing) set of parameters for Random Forest.

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 3871.00 | 5039.17 | 0.9481 |
| **Random Forest (tuned)** | **1380.57** | **2980.19** | **0.9818** |
| XGBoost (tuned) | 1475.26 | 3131.96 | 0.9799 |

*(Model comparison bar chart and scatter plots to be added here)*

## Results

- **Final model: Random Forest (Tuned)** — lowest error, highest explanatory power.
- Feature importance analysis showed `lag_1` dominating the model (~85% importance). However, removing `lag_1` only dropped performance by ~0.5% — indicating the model isn't overly reliant on a single signal, but has genuinely learned the underlying autocorrelation structure of sales.
- Predictions on the test set were visualized over time; the model successfully anticipated the sales surge during holiday periods (Nov-Dec 2012) it had never seen before.

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

Run the notebook sections in order: data loading → EDA → feature engineering → modeling → test predictions.

## Key Takeaways

- Random splitting and random K-Fold cross-validation can cause serious data leakage in time-series data — using `TimeSeriesSplit` made a real, measurable difference in model selection.
- XGBoost doesn't always outperform Random Forest; model choice depends on the structure of the dataset (particularly the presence of strongly autocorrelated features).
- Understanding *why* data is missing (as with MarkDown vs. CPI/Unemployment) is essential before deciding how to handle it.
