from flask import Flask, render_template, request
import pickle
import re
import string


app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorization.pkl", "rb"))

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        cleaned = clean_text(news)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        result = "Real" if prediction == 1 else "Fake"
        return render_template('index.html', prediction=result, input_text=news)

if __name__ == '__main__':
    app.run(debug=True)
