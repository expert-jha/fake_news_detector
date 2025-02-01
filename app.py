import tensorflow as tf
import streamlit as st
import joblib
import numpy
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

model = tf.keras.models.load_model("fake_news_detector.h5")

token = joblib.load("tokenizer.pkl")

st.title("Fake News Predictor")
text_input = st.text_area("Enter the news article")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

text_input = clean_text(text_input)

if st.button("Predict"):
    if text_input:
        text_vec = numpy.array(token.texts_to_sequences([text_input]))

        prediction = model.predict(text_vec)

        if prediction >= 0.7:
            st.write("Prediction : **Fake News**")
        else:
            st.write("Prediction : **Real News**")
    else:
        st.write("please write some text")
