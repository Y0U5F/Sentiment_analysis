import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from flask import Flask, request, render_template
import pandas as pd

# تهيئة NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# تحميل النموذج وVectoriser
model = joblib.load('LRmodel.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# إعداد قائمة كلمات التوقف
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']
STOPWORDS = set(stopwordlist)
punctuations_list = string.punctuation
lemmatizer = WordNetLemmatizer()

# دالة تنظيف النصوص
def clean_text(text):
    # تحويل إلى حروف صغيرة
    text = text.lower()
    # إزالة الروابط
    text = re.sub(r'((www.[^s]+)|(https?://[^s]+))', ' ', text)
    # إزالة الإشارات والهاشتاغ
    text = re.sub(r'@\w+|\#\w+', '', text)
    # إزالة الأرقام
    text = re.sub(r'[0-9]+', '', text)
    # إزالة الرموز
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)
    # إزالة كلمات التوقف
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# إنشاء تطبيق Flask
app = Flask(__name__)

# الصفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    input_text = ""
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text:
            # تنظيف النص المدخل
            cleaned_text = clean_text(input_text)
            # تحويل النص إلى متجه TF-IDF
            text_vector = vectorizer.transform([cleaned_text])
            # التنبؤ بالمشاعر
            prediction = model.predict(text_vector)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
    return render_template('index.html', sentiment=sentiment, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)