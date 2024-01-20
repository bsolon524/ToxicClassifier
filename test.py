import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Function to generate batches for prediction
def generate_batches(X, batch_size=32):
    total_size = X.shape[0]
    for start in range(0, total_size, batch_size):
        end = start + batch_size
        X_batch = X[start:end].toarray()  # Convert sparse matrix to dense
        yield X_batch

# Load the trained model and the vectorizer
model = tf.keras.models.load_model('toxic_comment_classifier.keras')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the test data and labels
test_data = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')

# Merge test data and labels on 'id'
merged_test_data = pd.merge(test_data, test_labels, on='id')

# Filter out rows where the toxicity labels are -1
merged_test_data = merged_test_data[merged_test_data['toxic'] != -1]

# Preprocess the test data
X_test = vectorizer.transform(merged_test_data['comment_text'])

# Extract the labels
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_test = merged_test_data[label_columns]

# Predict in batches
batch_size = 32
predictions = np.vstack([model.predict(batch) for batch in generate_batches(X_test, batch_size)])

# Process predictions to binary format
threshold = 0.5
binary_predictions = np.where(predictions > threshold, 1, 0)

# Evaluate predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

for i, label in enumerate(label_columns):
    print(f"Metrics for: {label}")
    print(f"Accuracy: {accuracy_score(y_test[label], binary_predictions[:, i])}")
    print(f"Precision: {precision_score(y_test[label], binary_predictions[:, i])}")
    print(f"Recall: {recall_score(y_test[label], binary_predictions[:, i])}")
    print(f"F1 Score: {f1_score(y_test[label], binary_predictions[:, i])}")
    print("\n")
