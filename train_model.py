import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Sample dataset (you can replace this with a real one)
data = {
    'text': [
        "I love this product!",
        "This is the worst thing ever.",
        "I'm not sure how I feel about this.",
        "Absolutely fantastic experience.",
        "Totally disappointed with the service.",
        "Not bad, could be better.",
        "I’m very happy with the quality.",
        "I hate the taste of this item.",
        "It's okay, nothing special.",
        "Great value for money!"
    ],
    'label': [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'neutral', 'positive'
    ]
}

df = pd.DataFrame(data)

# Preprocessing function
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', str(text))  # Remove non-word characters
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")
