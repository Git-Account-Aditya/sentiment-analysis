import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Fixed import


# Load the trained model
model_path = r"C:\Users\Aditya Rawat\Desktop\Sentiment Analysis\sentiment_analysis.keras"
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()  # Exit script if model loading fails


# Get the IMDb word index dictionary
word_index = imdb.get_word_index()

# Adjust word index for special tokens
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Reverse mapping for decoding sequences
reverse_word_index = {v: k for k, v in word_index.items()}

# Tokenizer setup
tokenizer = Tokenizer(num_words=20000, oov_token="<UNK>")
tokenizer.word_index = word_index

st.title("Sentiment Analysis on Movie Reviews")

text = st.text_area("Enter a movie review to analyze its sentiment.")

text = [text]

if st.button("Analyze"):
    decoded_review = " ".join([reverse_word_index.get(i, "<UNK>") for i in text])
    # print(f"\n🔍 Decoded Review:\n{decoded_review}\n")

    # Tokenize and pad the input review
    tokenized_seq = tokenizer.texts_to_sequences([decoded_review])
    tokenized_seq = pad_sequences(tokenized_seq, padding='post', maxlen=250)

    # Predict sentiment
    prediction = model.predict(tokenized_seq)[0][0]  # Extract probability
    good, bad = prediction, 1 - prediction

    # Print sentiment result
    # actual_sentiment = "Positive" if y_test[test_index] == 1 else "Negative"
    predicted_sentiment = "Positive" if good > bad else "Negative"

    # print(f"🎭 **Actual Sentiment:** {actual_sentiment}")
    st.write(f"🤖 **Predicted Sentiment:** {predicted_sentiment}")
    st.write(f"📊 Confidence: {good * 100:.2f}% Positive | {bad * 100:.2f}% Negative")