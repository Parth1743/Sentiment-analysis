# **Sentiment Analysis Using IMDB Dataset**

This project implements a **Sentiment Analysis Model** trained on the IMDB movie review dataset. The primary goal is to classify reviews as either **Positive** or **Negative** based on their content using Natural Language Processing (NLP) techniques and machine learning.

---

## **Features**
- **Dataset**: IMDB Dataset of 50K Movie Reviews ([Dataset Resource](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)).
- **Data Preprocessing**:
  - Cleaned text (removing URLs, special characters, and stopwords).
  - Tokenization and conversion to lowercase.
  - Vectorization using **TF-IDF** (Term Frequency-Inverse Document Frequency).
- **Model**: **Multinomial Naive Bayes**, a robust algorithm for text classification.
- **Evaluation**: Accuracy and classification report for precision, recall, and F1-score.

---

## **Results**
The model achieved the following performance:
- **Accuracy**: Between **85% and 90%** on the test data.
- **Precision**: Reliable classification of both positive and negative sentiments.
- **Recall**: High for positive reviews, with minor room for improvement in negative reviews.
- **F1-Score**: Demonstrates a good balance between precision and recall.

These results make the model suitable for analyzing large-scale movie reviews with reasonable accuracy.

---

## **Installation**
Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Parth1743/Sentiment-analysis.git
   cd Sentiment-analysis
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download) and save it in the project directory.

4. **Run the Script**:
   Execute the main script to train the model:
   ```bash
   python main.py
   ```

---

## **Project Workflow**
1. **Data Loading**:
   - Load the IMDB dataset and map the sentiment labels (positive = 1, negative = 0).

2. **Data Preprocessing**:
   - Clean text by removing noise and stopwords.
   - Use TF-IDF vectorization to convert textual data into numerical format.

3. **Model Training**:
   - Train a **Multinomial Naive Bayes** model on the preprocessed data.

4. **Model Evaluation**:
   - Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.

5. **Save Model**:
   - Save the trained model and vectorizer using `joblib` for future use.

6. **Visualization**:
   - Plot the sentiment distribution in the dataset using Matplotlib and Seaborn.

---

## **How to Use**
1. **Test Pre-Trained Model**:
   Use the pre-trained model (`pre-trained/sentiment_model.pkl`) and vectorizer (`pre-trained/vectorizer.pkl`) to test custom reviews:
   ```python
   import joblib
   from your_script import clean_text  # Replace with your script name

   # Load the model and vectorizer
   model = joblib.load('pre-trained/sentiment_model.pkl')
   vectorizer = joblib.load('pre-trained/vectorizer.pkl')

   # Test a custom review
   review = "The movie was an absolute masterpiece!"
   cleaned_review = clean_text(review)
   vectorized_review = vectorizer.transform([cleaned_review])
   prediction = model.predict(vectorized_review)
   print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
   ```

2. **Train the Model Again**:
   Modify and retrain the model using your custom data or parameters by running the main script.

---

## **Technologies Used**
- **Languages**: Python
- **Libraries**:
  - Pandas and NumPy for data manipulation.
  - Scikit-learn for machine learning.
  - NLTK for text preprocessing.
  - Matplotlib and Seaborn for data visualization.
- **Tools**: Joblib for saving and loading models.

---

## **Folder Structure**
```
Sentiment-analysis/
│
├── main.py                # Main script for training and evaluation
├── requirements.txt       # Dependencies
├── IMDB Dataset.csv       # Dataset file
├── pre-trained/           # Folder for pre-trained models
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
├── README.md              # Project documentation
```

---

## **Future Improvements**
- Incorporate deep learning models like LSTM or BERT for improved accuracy.
- Add support for neutral sentiment classification.
- Create a user-friendly GUI for real-time sentiment analysis.

---

## **License**
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## **Author**
**Parth Garg**  
[LinkedIn](http://www.linkedin.com/in/parth-garg-946227256) | [GitHub](https://github.com/Parth1743)
