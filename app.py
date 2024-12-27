import streamlit as st
import pickle
from joblib import load
import pandas as pd
from datetime import datetime

# Load the trained model and vectorizer
# with open('mnb.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

model = load("mnb.joblib")

# with open('vectorizer_mnb.pkl', 'rb') as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

vectorizer = load('vectorizer.joblib')

# Define CSV file paths
feedback_file = "chatbot_feedback.csv"
sentiment_file = "ticket_feedback.csv"

# Initialize CSV files if they do not exist
for file_path, columns in [(feedback_file, ["Timestamp", "Feedback"]),
                           (sentiment_file, ["Timestamp", "Ticket Description", "Predicted Sentiment"])]:
    try:
        pd.read_csv(file_path)
    except FileNotFoundError:
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)

# Define a function to make predictions
def predict_sentiment(input_text):
    # Transform the input text using the loaded vectorizer
    transformed_text = vectorizer.transform([input_text])
    sentiment_score = model.predict(transformed_text)[0]

    # Map sentiment score to labels
    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    sentiment_label = sentiment_labels[sentiment_score]
    return sentiment_score, sentiment_label

# Streamlit app UI
st.title("Customer Support Sentiment Analysis")

st.header("Predict the Sentiment of a Customer Ticket")

# Input box for user text
user_input = st.text_area("Enter Ticket Description:", placeholder="Type your ticket description here...")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip():  # Check if input is not empty
        sentiment_score, sentiment_label = predict_sentiment(user_input)

        # Log the ticket description and sentiment to the CSV file
        ticket_data = {"Timestamp": [datetime.now()], 
                       "Ticket Description": [user_input], 
                       "Predicted Sentiment": [sentiment_label]}
        ticket_df = pd.DataFrame(ticket_data)
        ticket_df.to_csv(sentiment_file, mode="a", index=False, header=False)

        st.write(f"**Input Ticket Description:** {user_input}")
        st.write(f"**Predicted Sentiment Score:** {sentiment_score}")
        st.write(f"**Predicted Sentiment Label:** {sentiment_label}")
    # else:
    #     st.error("Please enter a valid ticket description!")
        st.session_state["ticket_description"] = ""
    else:
        st.error("Please enter a valid ticket description!")


# Sidebar for user feedback
st.sidebar.title("Feedback")
feedback = st.sidebar.text_area("Any feedback for us? How's our Chatbot working?")
if st.sidebar.button("Submit Feedback"):
    if feedback.strip():  # Check if feedback is not empty
        # Log the feedback to the CSV file
        feedback_data = {"Timestamp": [datetime.now()], "Feedback": [feedback]}
        feedback_df = pd.DataFrame(feedback_data)
        feedback_df.to_csv(feedback_file, mode="a", index=False, header=False)
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Please provide valid feedback!")
