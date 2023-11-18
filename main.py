import streamlit as st
from model1 import model1
from model2 import model2

def main():
    st.title("Text Classification App")

    # Model selection
    model_selection = st.selectbox("Select Model", ["Model 1", "Model 2"])

    if model_selection == "Model 1":
        model2()
    elif model_selection == "Model 2":
        model1()





if __name__ == "__main__":
    main()
