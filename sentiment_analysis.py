import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample film reviews
reviews = [
    "The movie was absolutely fantastic with brilliant acting",
    "I hated the film, it was boring and too long",
    "Amazing storyline and great direction",
    "The movie was terrible and disappointing",
    "It was a wonderful cinematic experience"
]

# Stopwords
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)

# Analyze sentiment
for review in reviews:
    clean_review = preprocess(review)
    polarity = TextBlob(clean_review).sentiment.polarity
    
    sentiment = "Positive" if polarity > 0 else "Negative"
    
    print("Review:", review)
    print("Processed:", clean_review)
    print("Sentiment:", sentiment)
    print("-" * 50)