import joblib

# Load the trained model and vectorizer
model = joblib.load('optimized_sentiment_model.pkl')
vectorizer = joblib.load('optimized_vectorizer.pkl')

# Define a text cleaning function (reuse from main.py)
import re
from nltk.corpus import stopwords

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)     # Remove special characters
    text = text.lower()                 # Convert to lowercase
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Function to predict sentiment
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])  # Transform input text
    prediction = model.predict(vectorized_text)             # Predict sentiment
    return "Positive" if prediction == 1 else "Negative"

# Test the model
print(predict_sentiment("I hate this movie"))  # Example 1
print(predict_sentiment("This is the best experience ever."))  # Example 2
