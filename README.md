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
