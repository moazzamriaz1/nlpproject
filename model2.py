import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def model2():
    # Load the saved model
    model_filename = "emotion_model.joblib"
    loaded_model = joblib.load(model_filename)

    # Load the TfidfVectorizer (assuming you used TfidfVectorizer during training)
    vectorizer_filename = "count_vectorizer.joblib"  # Update this to the correct filename
    vectorizer = joblib.load(vectorizer_filename)

    # Streamlit App
    st.title("Emotion Prediction App")

    # Input text from the user
    user_input = st.text_area("Enter your text:")

    # Analyze button
    if st.button("Analyze"):
        # Make predictions with new data
        if user_input:
            new_data = [user_input]
            new_features = vectorizer.transform(new_data)
            new_predictions = loaded_model.predict_proba(new_features)

            # Display predictions using a progress bar
            st.subheader("Emotion Scores:")

            # Assuming there are three classes (Fear, Anger, Joy)
            progress_bar_fear = st.progress(new_predictions[0][0])
            st.write("Fear:", round(new_predictions[0][0], 2))
            progress_bar_anger = st.progress(new_predictions[0][1])
            st.write("Anger:", round(new_predictions[0][1], 2))
            progress_bar_joy = st.progress(new_predictions[0][2])
            st.write("Joy:", round(new_predictions[0][2], 2))





# Call the function to run the app
if __name__ == "__main__":
    model2()
