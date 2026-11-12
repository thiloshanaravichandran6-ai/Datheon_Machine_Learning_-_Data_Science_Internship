# Task 2: California Housing Price Prediction (OLS vs Ridge vs Lasso)

## Objective
Predict median house values using the California Housing dataset. The task involves engineering interaction features, imputing missing values, handling multicollinearity, and comparing Ordinary Least Squares (OLS), Ridge (L2) and Lasso (L1) regression using RMSE, MAE and R².

## Dataset
- Source: [California Housing dataset](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
- 20,640 rows, 10 columns
- Target: `median_house_value`
- Categorical feature: `ocean_proximity`

## Approach
1. **Load data** directly from the URL.
2. **Impute missing values:** `total_bedrooms` (207 nulls) filled with the median, fitted on the training set only.
3. **Correlation heatmap and VIF** to detect multicollinearity.
4. **Feature engineering:** rooms per household, bedrooms per room, population per household, and interaction terms (income × rooms, income × age, latitude × longitude).
5. **Preprocessing:** train/test split (80/20), standard scaling of numeric features, one-hot encoding of `ocean_proximity`, all inside a scikit-learn pipeline.
6. **Models:** OLS, Ridge and Lasso, with Ridge and Lasso tuned by `GridSearchCV` (5-fold CV) over a range of alpha values.
7. **Evaluation:** RMSE, MAE and R² on the test set.
8. **Plots:** predicted vs actual values and residual distributions.

## Results (test set)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| OLS   | ~68,500 | ~49,200 | ~0.64 |
| Ridge | ~68,500 | ~49,200 | ~0.64 |
| Lasso | ~68,700 | ~49,300 | ~0.64 |

## Key Findings
- All three models perform almost identically; OLS is marginally best.
- GridSearchCV selected the smallest alphas in the grids, so little regularization was needed (train R² ≈ test R², so no overfitting).
- Multicollinearity is present (high VIF for latitude/longitude, room and bedroom counts and the interaction terms). It affects coefficient stability more than prediction accuracy.
- Residuals fan out as predictions increase and pile up at the $500k cap in the target, which is a limitation of linear models.

## Next Steps
- Log-transform the target variable.
- Try non-linear models such as Random Forest or Gradient Boosting.

## Tools
Python, pandas, NumPy, scikit-learn (LinearRegression, Ridge, Lasso, GridSearchCV), statsmodels, matplotlib, seaborn

## Files
- `California_Housing_Regression.ipynb`: the full regression notebook with metrics and plots
