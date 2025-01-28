import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Download NLTK resources only once
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load dataset
data = pd.read_csv(r'path to the csv file or url')

# Display the first few rows
print(data.head())

# Define a text cleaning function
stop_words = set(stopwords.words('english'))  # Load stopwords once

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize words
    text = ' '.join([word for word in tokens if word not in stop_words])  # Remove stopwords
    return text

# Map sentiment to numeric values and clean text
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
data['cleaned_text'] = data['review'].apply(clean_text)

# Check for missing values
print(data.isnull().sum())

# Plot sentiment distribution
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"])
plt.show()

# Generate WordCloud for Word Frequency
all_words = ' '.join(data['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_words)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Reviews", fontsize=16)
plt.show()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['sentiment'], test_size=0.2, random_state=42)

# Optimize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))  # Fewer features and bigrams for better performance

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, 'optimized_sentiment_model.pkl')  # Save the model
joblib.dump(vectorizer, 'optimized_vectorizer.pkl')  # Save the vectorizer
print("Optimized model and vectorizer saved successfully.")

# Plot Metrics (Precision, Recall, F1-score) for Positive and Negative Classes
metrics_df = pd.DataFrame(report).transpose()
metrics_df = metrics_df[['precision', 'recall', 'f1-score']].iloc[:2]  # Only for classes 0 (Negative) and 1 (Positive)

metrics_df.plot(kind='bar', figsize=(8, 6), colormap='viridis')
plt.title("Model Evaluation Metrics", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"], rotation=0)
plt.legend(loc='lower right')
plt.ylim(0, 1)
plt.show()

# Confusion Matrix Visualization
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Negative", "Positive"], cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
