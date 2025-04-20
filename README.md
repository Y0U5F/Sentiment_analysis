# Sentiment Analysis for Twitter

A Python-based project for analyzing the sentiment of Twitter tweets, classifying them as positive or negative using NLP and Machine Learning techniques. The project uses the Sentiment140 dataset and includes a Flask web interface for real-time sentiment prediction.

## Features
- Preprocesses text data using NLTK (tokenization, lemmatization, stopword removal).
- Trains machine learning models (Logistic Regression, SVM, Naive Bayes) with TF-IDF vectorization.
- Provides a Flask web app to input text (word, sentence, or article) and predict sentiment.
- Evaluates models using accuracy, F1-score, and ROC-AUC.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Y0U5F/Sentiment_Analysis.git
Install dependencies:
bash

pip install -r requirements.txt
Download NLTK data:
python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Run the Flask app:
bash

python app.py
Open http://127.0.0.1:5000/ in your browser.
Requirements
See requirements.txt for the full list of dependencies.

Dataset
The project uses the Sentiment140 dataset with 1.6M tweets.

License
This project is licensed under the MIT License - see the  file for details.
