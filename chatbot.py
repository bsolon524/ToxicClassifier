import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load the saved model and vectorizer
model = load_model('toxic_comment_classifier.keras')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def classify_text(text):
    vectorized_text = vectorizer.transform([text]).toarray()
    predictions = model.predict(vectorized_text)[0]  # Get the first (and only) prediction
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return {label: (pred > 0.5) for label, pred in zip(labels, predictions)}

def chatbot_response(user_input):
    prediction = classify_text(user_input)
    toxic_types = [label for label, is_toxic in prediction.items() if is_toxic]
    
    if toxic_types:
        response = f"Harmful sentiment detected: {', '.join(toxic_types)}"
    else:
        response = "Message received"
    
    return response
def run_chatbot():
    print("Chatbot activated. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    run_chatbot()