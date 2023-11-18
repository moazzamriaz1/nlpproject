import streamlit as st
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import pandas as pd

def model1():
    # Your Model 1 code here
    st.subheader("Model 2 Analysis")
    model_path = "mymodel.pth"  # Replace with the actual path to your trained model
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    # Labels for your specific task
    labels = ["anger", "annoyance", "neutral", "disgust", "sadness",
              "fear", "caring",
              "love", "joy"]  # Replace with your actual label names

    # Streamlit app
    user_input = st.text_area("Enter text for analysis:")
    if st.button("Analyze"):
        if user_input:
            # Tokenize and preprocess the input
            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            # Make prediction
            with torch.no_grad():
                output = model(input_ids)
            # Get predicted probabilities
            probabilities = torch.sigmoid(output.logits)

            # Check if the lengths match before creating the DataFrame
            if len(labels) == len(probabilities[0]):
                # Display the probabilities as individual bars
                df = pd.DataFrame({
                    "Label": labels,
                    "Probability": probabilities[0].tolist()
                })

                st.bar_chart(df.set_index("Label"))

                # Display the emotion labels and scores
                st.subheader("Emotion Analysis Output:")
                for i, result in enumerate(sorted(zip(labels, probabilities[0]), key=lambda x: x[1], reverse=True)):
                    label, score = result
                    st.write(f"{i + 1}. {label.capitalize()}: {score:.4f}")
            else:
                st.error("Error: The length of labels and probabilities does not match.")
        else:
            st.warning("Please enter text for analysis.")