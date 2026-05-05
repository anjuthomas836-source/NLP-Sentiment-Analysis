# NLP Sentiment Analysis - Amazon Reviews

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download (run only first time)
nltk.download('punkt')
nltk.download('stopwords')

# Dataset
reviews = [
    "This product is amazing and works perfectly",
    "Worst purchase ever, completely useless",
    "Very good quality, I am satisfied",
    "Terrible experience, broke after one use",
    "Excellent value for money, highly recommend",
    "Not worth the price, very disappointing",
    "I love it, will buy again",
    "Bad packaging and poor quality",
    "Superb performance, exceeded expectations",
    "Waste of money, do not buy"
]

# Stopwords
stop_words = set(stopwords.words('english'))

print("\n===== SENTIMENT ANALYSIS OUTPUT =====\n")

for review in reviews:
    # Tokenization
    tokens = word_tokenize(review.lower())

    # Remove stopwords
    filtered = [word for word in tokens if word not in stop_words]

    # Join words
    processed_text = " ".join(filtered)

    # Sentiment analysis
    blob = TextBlob(processed_text)
    polarity = blob.sentiment.polarity

    # Classification
    if polarity > 0:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print("Original Review :", review)
    print("Processed Text  :", processed_text)
    print("Sentiment       :", sentiment)
    print("-----------------------------------")