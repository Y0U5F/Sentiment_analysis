# Sentiment Analysis for Twitter

A Python-based project for analyzing the sentiment of Twitter tweets, classifying them as positive or negative using Natural Language Processing (NLP) and Machine Learning techniques. The project leverages the Sentiment140 dataset and includes a Flask web interface for real-time sentiment prediction.

## Features

- Preprocesses tweet text using NLTK (tokenization, lemmatization, stopword removal).
- Trains machine learning models (Logistic Regression, SVM, Naive Bayes) with TF-IDF vectorization.
- Provides a Flask web app to input text (word, sentence, or article) and predict sentiment.
- Evaluates model performance using accuracy, F1-score, and ROC-AUC.

## Installation

Follow these steps to set up and run the project locally:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Y0U5F/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Run the Flask app**:

   ```bash
   python src/app.py
   ```

5. **Open the app**: Open your browser and navigate to:

   ```
   http://127.0.0.1:5000/
   ```

## Requirements

All dependencies are listed in the `requirements.txt` file. Key libraries include:

- Flask
- NumPy (&lt;2.0)
- Pandas
- Scikit-learn
- NLTK
- Joblib

## Dataset

The project uses the Sentiment140 dataset, which contains 1.6 million tweets labeled as positive or negative.

## Project Structure

```
Sentiment_Analysis/
├── src/
│   └── app.py              # Flask application
├── templates/
│   └── index.html          # HTML template for the web interface
├── models/
│   ├── LRmodel.pkl         # Trained Logistic Regression model
│   └── vectorizer.pkl      # TF-IDF Vectorizer
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
└── LICENSE                 # MIT License
```

## Usage

- Access the web interface at `http://127.0.0.1:5000/`.
- Enter a word, sentence, or article in the text box and click "Analyze" to predict sentiment (Positive or Negative).
- ![image](https://github.com/user-attachments/assets/f97f771c-a2fa-4110-8a7d-be642e66a294)
- Example inputs:
  - Word: "awesome" → Positive
  - Sentence: "I hate this product" → Negative
  - Article: "This movie was fantastic! The acting was great..." → Positive

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Sentiment140 for providing the dataset.
- Flask for the web framework.
- Scikit-learn and NLTK for machine learning and NLP tools.
