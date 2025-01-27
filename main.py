import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data = pd.read_csv(r'C:\Users\Parth garg\Documents\GitHub\Sentiment-analysis\IMDB Dataset.csv')

# Check the first few rows
print(data.head())

# Define a text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)     # Remove special characters
    text = text.lower()                 # Convert to lowercase
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(text)

# Map sentiment to numeric values
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Apply text cleaning
data['cleaned_text'] = data['review'].apply(clean_text)

# Check for NaN or missing values
print(data.isnull().sum())  # Ensure no missing values

# Check the distribution of sentiments
sentiment_counts = data['sentiment'].value_counts()

# Bar plot for sentiment distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"])
plt.show()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['sentiment'], test_size=0.2, random_state=42)

# Initialize a vectorizer (TF-IDF recommended)
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data, and transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')  # Save the model
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save the vectorizer

print("Model and vectorizer saved successfully.")
