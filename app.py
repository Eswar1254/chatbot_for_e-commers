import streamlit as st
import json
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load intents and BERT model
intents = json.loads(open('intents (1).json').read())
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intents['intents']))

# Function to preprocess text and get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

# Function to predict intent using BERT embeddings
def predict_intent(text):
    embeddings = get_bert_embeddings(text)
    predicted_class = torch.argmax(embeddings, dim=1).item()
    return predicted_class

# Function to get response based on predicted intent
def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            responses = intent['responses']
            return random.choice(responses)
    return "Sorry, I didn't understand that."

# Function to handle user input and get chatbot response
def chatbot_response(user_input):
    # Preprocess user input
    user_input = user_input.strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    # Predict intent
    intent_idx = predict_intent(user_input)
    intent_tag = intents['intents'][intent_idx]['tag']
    # Get response based on intent
    response = get_response(intent_tag)
    return response

# Streamlit app
st.title("e-commerce Chatbot")

user_input = st.text_input("You: ")
if st.button("Send"):
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100)  
