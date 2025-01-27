import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['sentiment'], test_size=0.2, random_state=42)

# Initialize a vectorizer (TF-IDF recommended)
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data, and transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Convert to dense arrays for TensorFlow
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# Define the neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Make predictions
y_pred = (model.predict(X_test_tfidf) > 0.5).astype("int32")

# Classification report
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
model.save('sentiment_model.h5')  # Save the model
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save the vectorizer

print("Model and vectorizer saved successfully.")