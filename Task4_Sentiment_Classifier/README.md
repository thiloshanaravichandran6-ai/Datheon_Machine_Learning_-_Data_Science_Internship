# Sentiment Analysis Classifier (TF-IDF + Naive Bayes / Logistic Regression)

A natural language processing pipeline that classifies customer product and movie reviews as **positive** or **negative**. It cleans raw text, converts it to TF-IDF features, trains two classic machine learning models, compares them, and provides a helper function for testing your own phrases.

## Project Overview

| Item | Details |
|---|---|
| Task | Binary sentiment classification (positive / negative) |
| Dataset | `sentiment_reviews.csv` (`review_id`, `review_text`, `sentiment`) |
| Features | TF-IDF, `max_features=5000`, unigrams + bigrams |
| Models | Multinomial Naive Bayes, Logistic Regression |
| Libraries | scikit-learn, NLTK, pandas, NumPy, matplotlib, seaborn |
| Deliverable | `sentiment_classifier.ipynb` |

## Pipeline

```
Raw review text
   │
   ▼
Cleaning ─ remove HTML tags, lowercase, expand "n't", strip punctuation/digits
   │
   ▼
Tokenization ─ split into words
   │
   ▼
Stopword removal ─ NLTK English list (negation words kept)
   │
   ▼
Lemmatization ─ WordNet ("stopped" → "stop", "batteries" → "battery")
   │
   ▼
TF-IDF vectorization ─ TfidfVectorizer(max_features=5000)
   │
   ▼
Train: MultinomialNB  +  LogisticRegression
   │
   ▼
Evaluate: accuracy, F1, confusion matrix, cross-validation
   │
   ▼
predict_sentiment("your text")
```

## Getting Started

### 1. Requirements

- Python 3.9 or newer
- Jupyter Notebook, JupyterLab, or Google Colab

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
# optional, for the button-style tester widget
pip install ipywidgets
```

### 3. Get the dataset

The notebook downloads the CSV automatically from:

```
https://www.datheon.in/datasets/sentiment_reviews.csv
```

If the download fails (no internet, firewall), save the file manually as `sentiment_reviews.csv` in the same folder as the notebook. The notebook falls back to this local copy.

### 4. Run the notebook

```bash
jupyter notebook sentiment_classifier.ipynb
```

Then choose **Kernel → Restart & Run All**. The first run downloads small NLTK resources (stopwords, WordNet). If those downloads are blocked, the notebook falls back to scikit-learn's stopword list and skips lemmatization.

## Using the Prediction Helper

After running all cells, call `predict_sentiment()` with any text:

```python
predict_sentiment("Absolutely fantastic product, works great!")
# 😊 POSITIVE (87%)  [lr]  ->  Absolutely fantastic product, works great!

predict_sentiment("Not worth the money, very disappointed.", model="nb")
# 😞 NEGATIVE (74%)  [nb]  ->  Not worth the money, very disappointed.
```

| Parameter | Description |
|---|---|
| `text` | The raw review string |
| `model` | `"lr"` for Logistic Regression (default) or `"nb"` for Naive Bayes |

It returns a dictionary: `{"text": ..., "label": "positive" | "negative", "confidence": 0.0-1.0}`.

(The confidence values above are illustrative; your numbers will vary with the data.)

**Interactive mode:** uncomment `interactive_test()` in the notebook to type phrases one by one (`quit` to exit), or use the ipywidgets text box and Analyze button if `ipywidgets` is installed.

## Evaluation

The notebook reports:

- Train and test accuracy
- Macro F1 score
- Precision / recall classification report
- Confusion matrices for both models
- Stratified k-fold cross-validation (mean ± standard deviation)
- The most positive and most negative words according to Logistic Regression

### Important note on dataset size

The provided CSV is very small (about 10 sample rows, 9 after neutral reviews are removed), so accuracy on it has high variance and should not be treated as a reliable benchmark. To get a more trustworthy comparison, the final section of the notebook repeats the experiment on NLTK's 2,000-review `movie_reviews` corpus. It needs internet access to download the corpus and is skipped automatically if unavailable.

In a sample run on that larger corpus, Logistic Regression scored slightly higher than Naive Bayes (roughly 85% vs 83% test accuracy). Your exact figures may differ.

## Design Decisions

- **Neutral reviews are removed.** The task is binary, so rows labelled `neutral` are filtered out.
- **Negations are kept.** Words like `not`, `no`, and `never` are excluded from the stopword list. Removing them would turn "not worth the money" into "worth money" and flip the sentiment.
- **Bigrams are included** (`ngram_range=(1, 2)`) so phrases such as "not worth" and "stop work" are captured.
- **No data leakage.** The TF-IDF vectorizer is fit on training data only, and inside a pipeline during cross-validation.
- **Final models are refit on all data** after evaluation, before being used in `predict_sentiment()`.

## Project Structure

```
.
├── sentiment_classifier.ipynb   # Full pipeline, benchmarks, and interactive tester
├── sentiment_reviews.csv        # Dataset (auto-downloaded, or place manually)
└── README.md                    # This file
```

## Limitations

- Small training data limits accuracy and confidence calibration.
- Bag-of-words models do not understand sarcasm, context, or complex negation ("I expected it to be terrible, but it was great").
- Phrases that use words never seen in training are classified from the remaining known words only.
- Neutral sentiment is not supported.

## Possible Improvements

- Train on a larger labelled dataset (IMDB, Amazon reviews).
- Add a third `neutral` class for three-way classification.
- Tune hyperparameters (`alpha` for Naive Bayes, `C` for Logistic Regression) with `GridSearchCV`.
- Try SpaCy for tokenization and lemmatization.
- Compare against other models (Linear SVM, Random Forest) or a transformer such as DistilBERT.
- Wrap `predict_sentiment()` in a simple web app with Streamlit or Flask.

## License

This project was created for educational purposes as part of an internship assignment.

