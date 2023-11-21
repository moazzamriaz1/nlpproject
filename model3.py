import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


def model3():
    # Load the CSV file
    df = pd.read_csv('Emotion_classify_Data.csv')

    # Assuming your CSV has two columns: 'text' and 'label'
    X = df['Comment']
    y = df['Emotion']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a text classification pipeline using a bag-of-words model and a Naive Bayes classifier
    model = make_pipeline(CountVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Function to make predictions
    def predict_emotion(text):
        prediction = model.predict([text])
        return prediction[0]

    # Streamlit app
    

    # User input for prediction
    user_input = st.text_area("Enter a sentence:")

    if st.button("Predict"):
        if user_input:
            # Make prediction
            prediction = predict_emotion(user_input)
            st.success(f"Predicted Emotion: {prediction}")
        else:
            st.warning("Please enter a sentence for prediction.")

