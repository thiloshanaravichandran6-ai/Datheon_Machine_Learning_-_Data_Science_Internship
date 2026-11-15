# Datheon Machine Learning & Data Science Internship

This repository contains my project work for the 4-week Machine Learning
& Data Science internship by Datheon.

## Tasks Completed

### Task 1: Multi-Class Classification Pipeline (Iris Dataset)
Built an end-to-end classification pipeline using Scikit-Learn to predict
iris flower species from petal and sepal measurements.

- Explored the data with Pandas and a Seaborn pairplot
- Split the data 80/20 into train and test sets (stratified)
- Scaled features with `StandardScaler` (fitted on training data only to avoid data leakage)
- Trained Logistic Regression and Decision Tree models
- Evaluated with accuracy, precision, recall, F1-score and confusion matrices
- Visualized decision boundaries using petal length and petal width

**Results**

| Model | Accuracy |
|---|---|
| Logistic Regression | 93.3% |
| Decision Tree | 96.7% |

Both models classified setosa perfectly. All errors were between versicolor
and virginica, whose measurements overlap. The accuracy gap is only one test
sample, so the models perform comparably.

Folder: [`Task1_Iris_Classification`](Task1_Iris_Classification)


### Task 2: California Housing Price Prediction

 Predict median house values and compare OLS, Ridge (L2) and Lasso (L1) regression using RMSE, MAE and R².


- Imputed missing values (`total_bedrooms`) with the median
- Checked multicollinearity with a correlation heatmap and VIF
- Engineered ratio and interaction features
- Scaled numeric features and one-hot encoded `ocean_proximity`
- Tuned Ridge and Lasso with `GridSearchCV` over multiple alphas
- Plotted predicted vs actual values and residual distributions

**Results:**

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| OLS   | ~68,500 | ~49,200 | ~0.64 |
| Ridge | ~68,500 | ~49,200 | ~0.64 |
| Lasso | ~68,700 | ~49,300 | ~0.64 |

All three models perform almost the same. Little regularization was needed, and the main limit is that linear models can't capture the non-linear patterns in the data.

Folder: [`Task2_California_Housing_Regression`](Task1_California_Housing_Regression)

### Task 3: Sentiment Analysis Classifier
 
Classify customer product and movie reviews as positive or negative using TF-IDF features, comparing Multinomial Naive Bayes and Logistic Regression.
 
- Removed HTML tags, punctuation and stopwords (negation words like "not" kept)
- Lowercased, tokenized and lemmatized the text
- Converted text to TF-IDF vectors (`max_features=5000`, unigrams + bigrams)
- Trained Multinomial Naive Bayes and Logistic Regression
- Evaluated with accuracy, F1, confusion matrix and cross-validation
- Built a `predict_sentiment()` helper and an interactive tester for custom phrases
**Results** (larger 2,000-review benchmark; the provided CSV is too small for reliable scores):
 
| Model | Accuracy | F1 (macro) |
|---|---|---|
| Multinomial Naive Bayes | ~83% | ~0.83 |
| Logistic Regression | ~85% | ~0.85 |
 
Both models perform similarly, with Logistic Regression slightly ahead. The provided dataset has only a handful of reviews, so its scores vary a lot, and the main limit is that bag-of-words models can't handle sarcasm or complex context.
 
Folder: [`Task3_Sentiment_Classifier`](Task3_Sentiment_Classifier)
 
