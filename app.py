# Import necessary functions or classes from the converted Python script
from utils import chatbot_response

import streamlit as st

# Streamlit app
st.title("Simple Chatbot")

user_input = st.text_input("You: ")
if st.button("Send"):
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100)
