#!/usr/bin/env python
# coding: utf-8

# In[7]:


# train_sentiment_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Dataset sederhana buatan
komentar = [
    "Saya sangat senang dengan layanan ini",       # Positif
    "Produk ini buruk dan mengecewakan",            # Negatif
    "Pengalaman yang menyenangkan",                 # Positif
    "Saya tidak suka produk ini",                   # Negatif
    "Layanan pelanggan sangat membantu",            # Positif
    "Kualitas produk sangat jelek",                 # Negatif
    "Ini adalah produk terbaik yang pernah saya beli", # Positif
    "Saya kecewa dan marah dengan pelayanan",       # Negatif
]

label = ['Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif']

# Buat pipeline: TF-IDF + Naive Bayes
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

model.fit(komentar, label)

# Simpan model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:


from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(np.array([[float(data['data'])]]))
    return jsonify({'result': prediction[0]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

