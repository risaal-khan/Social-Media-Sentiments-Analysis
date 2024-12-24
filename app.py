import streamlit as st
import joblib

model = joblib.load("PycharmProjects\Sentiment_Analysis\sentiment_analysis_model.pkl")
vectorizer = joblib.load(r"PycharmProjects\Sentiment_Analysis\vectorizer.pkl")

st.title("Sentiment Analysis App")
st.write("Enter some text below, and the app will predict its sentiment.")

user_input = st.text_area("Enter your text here", placeholder="Type your text...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        sentiment = prediction

        st.subheader("Prediction")
        st.write(f"The sentiment of the text is: **{sentiment}**")
    else:
        st.error("Please enter some text to analyze.")
