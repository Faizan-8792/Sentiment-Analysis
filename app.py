import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd 

# Custom CSS for background and footer
st.markdown("""
    <style>
    /* Gradient background for whole app */
    .stApp {
        background: linear-gradient(to bottom right, #f0f4f7, #dbe9f4, #e0c3fc, #8ec5fc);
        color: #000000;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 16px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        color: #000000;
    }

    .footer span {
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Mapping for emoji and colors
sentiment_emoji = {
    'positive': '😊',
    'negative': '😡',
    'neutral': '😐'
}

sentiment_color = {
    'positive': 'green',
    'negative': 'red',
    'neutral': 'orange'
}

# Streamlit Config
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")
st.title("📝 Sentiment Analysis App")
st.markdown("**Enter a sentence below to analyze the sentiment**")

# User input
user_input = st.text_area("✏️ Your text here:")

# Predict and show result
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned_input = preprocess(user_input)
        vector_input = vectorizer.transform([cleaned_input]).toarray()
        proba = model.predict_proba(vector_input)

        predicted_index = np.argmax(proba)
        prediction = model.classes_[predicted_index]
        confidence = round(proba[0][predicted_index] * 100, 2)
        emoji = sentiment_emoji[prediction]
        color = sentiment_color[prediction]

        st.markdown(f"### Sentiment: <span style='color:{color}'>{prediction.capitalize()} {emoji}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score:** {confidence}%")

        st.markdown("#### 🔎 Detailed Sentiment Probabilities")
        prob_df = pd.DataFrame({
            'Sentiment': model.classes_,
            'Probability (%)': [round(p * 100, 2) for p in proba[0]]
        })
        prob_df = prob_df.sort_values(by='Probability (%)', ascending=False)

        st.dataframe(
            prob_df.style.bar(subset=['Probability (%)'], color='lightgreen')
        )

# Footer
st.markdown("""
    <div class="footer">
        <span>Project made by</span>
        <span>- Saifullah Faizan</span>
        <span>- Shrishti Singh</span>
        <span>- Ahmed Jamal</span>
    </div>
""", unsafe_allow_html=True)
