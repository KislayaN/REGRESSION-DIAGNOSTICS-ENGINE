# Regression Diagnostics Toolkit
A modular regression diagnostics system that analyzes model behavior across multiple linear regression variants, focusing on performance, optimization, and statistical validity.

## Overview

This project evaluates and compares regression models beyond basic metrics. It provides insights into:

- model performance (MSE, R²)
- optimization behaviour (gradient descent convergence)  
-  statistical properties (residuals, multicollinearity)  
-  data sensitivity (scaling impact, feature influence)

## Problem

Standard regression workflows focus on prediction accuracy only.
They do not explain:

- how optimization behaves
- how scaling affects convergence
- how multicollinearity impacts coefficients
- whether model assumptions hold

This project addresses these gaps.

## Features

- Implements 5 models:
  
  * OLS (Normal Equation)
  * Gradient Descent OLS
  * Linear Regression
  * Ridge (L2)
  * Lasso (L1)

-  Provides:

   * MSE and R² comparison
   * Gradient descent convergence analysis
   * Residual diagnostics
   * Multicollinearity detection
   * Feature importance analysis
   * Scaling sensitivity analysis
   * Automated insight generation

## Architecture

Pipeline flow:

Data → Validation → Sanity Check → Preprocessing → Training → Evaluation → Analysis

Structure:

```
| analysis/
| assets/
| core/
|    analyzer.py
| data/
| evaluation_metrics/
| feature_scaling/
| helpers/
| models/
| train_test_split/
| visualization/
| main.py
```

## Usage
```
from core.analyzer import Analyzer
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
target = housing.frame.columns.to_list()[-1]

analyzer = Analyzer()
results = analyzer.run(data, target)
results = analyzer.plot() # for plotting 
```

## Accessing Results
#### Metrics
```
results["metrics"]["values"]["mse"] # mse values for all models
results["metrics"]["values"]["r2"] # r2 values for all models
```
#### Residuals
```
results["residuals"]["values"]['residuals'] # residual values
results["residuals"]["values"]['mean'] # residual mean
results["residuals"]["values"]['variance'] # residual variance
```
#### Convergence
```
results["convergence"]["values"]['cost-history-scaled']
results["convergence"]["values"]['cost-history-unscaled']
```
#### Feature importance
```
results["coefficients"]["values"]['scaled'] # feature importance for models trained over scaled data
results["coefficients"]["values"]['unscaled'] # feature importance for models trained over unscaled data
top_featuers = analyzer.strong_features # top features  
```
#### Insights
```
results["insights"]["summary"]
results["insights"]["detailed-insights"]
```
#### Example Insights

- Ridge achieves lower error under multicollinearity
- Scaling improves gradient descent convergence
- Residuals are centered but show variance differences
- Feature importance remains stable across scaling

#### Design Decisions

- Separation of concerns:
  * computation (metrics/)
  * interpretation (analysis/)
  * visualization (visualization/)
- Scaling applied only where required (GD, regularized models)
- Insight generation implemented using rule-based logic
- Pipeline designed for extensibility and modularity

#### Limitations

- Assumes numeric datasets
- No automatic handling of missing values
- No categorical encoding
- Correlation-based multicollinearity detection (VIF not implemented)

#### Future Improvements

- Add VIF for multicollinearity
- Support categorical features
- Add cross-validation
- Extend to non-linear models
- Build API / dashboard interface

#### What This Demonstrates

- Understanding of linear regression variants
- Optimization techniques (gradient descent behavior)
- Statistical diagnostics in ML
- Modular ML system design
- Ability to move beyond “train & predict”

#### Author
Kislaya N  
ML Engineer (aspiring)
