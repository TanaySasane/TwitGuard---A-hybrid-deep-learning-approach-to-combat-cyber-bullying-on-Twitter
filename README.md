# TwitGuard - A Hybrid Deep Learning Approach to Combat Cyber Bullying on Twitter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TanaySasane/TwitGuard---A-hybrid-deep-learning-approach-to-combat-cyber-bullying-on-Twitter/blob/main/Tweeter%20Cyberbullying.ipynb)

## Overview

TwitGuard is a machine learning project that detects and classifies cyberbullying in tweets. It analyzes tweet text and categorizes it into one of six classes using NLP preprocessing and classical ML classifiers.

## Dataset

- Source: [Cyberbullying Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- 47,692 tweets labeled across 6 categories:

| Label | Description |
|---|---|
| `religion` | Bullying based on religion |
| `age` | Bullying based on age |
| `ethnicity` | Bullying based on ethnicity/race |
| `gender` | Bullying based on gender |
| `other_cyberbullying` | Other forms of cyberbullying |
| `not_cyberbullying` | Not cyberbullying |

## Project Pipeline

```
Raw Tweets
    ↓
Text Preprocessing (emoji removal, decontraction, stopword removal, stemming, lemmatization)
    ↓
TF-IDF Vectorization
    ↓
Model Training (Logistic Regression / Naive Bayes / LinearSVC)
    ↓
Evaluation (Accuracy, F1 Score, Confusion Matrix)
    ↓
Save Model (model.pkl + tfidf_vectorizer.pkl)
```

## Text Preprocessing Steps

1. Remove emojis
2. Expand contractions (e.g. `don't` → `do not`)
3. Lowercase, remove URLs, mentions, numbers, punctuation
4. Remove stopwords
5. Clean hashtags
6. Remove special characters (`$`, `&`)
7. Porter Stemming
8. WordNet Lemmatization

## Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Multinomial Naive Bayes | Fast probabilistic classifier |
| LinearSVC | Best performing model (~83% accuracy) |

## Results

Best model: **LinearSVC**
- Accuracy: ~83%
- F1 Score (macro): ~82%

## How to Run

### On Google Colab (Recommended)
Click the **Open in Colab** badge above. Upload `cyberbullying_tweets.csv` when prompted and run all cells.

### Locally
```bash
git clone https://github.com/TanaySasane/TwitGuard---A-hybrid-deep-learning-approach-to-combat-cyber-bullying-on-Twitter.git
cd TwitGuard---A-hybrid-deep-learning-approach-to-combat-cyber-bullying-on-Twitter
pip install pandas numpy matplotlib seaborn plotly emoji nltk pillow wordcloud scikit-learn
jupyter notebook "Tweeter Cyberbullying.ipynb"
```

Place `cyberbullying_tweets.csv` in the same folder before running.

## Requirements

- Python 3.7+
- pandas, numpy
- scikit-learn
- nltk
- emoji
- matplotlib, seaborn, plotly
- wordcloud, pillow

## Output

After running the notebook, two files are saved:
- `model.pkl` — trained LinearSVC model
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer

These can be loaded for inference without retraining.

## Author

**Tanay Sasane**
