import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the data
data = pd.read_csv('train.csv')

# Data Preprocessing
X = data['comment_text']
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y = data[label_columns]

# Text Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# TensorFlow Model for Multi-Label Classification
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(label_columns), activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generator Function for Batch Processing
def generate_batches(X, y, batch_size=32):
    total_size = X.shape[0]
    while True:  # Loop indefinitely
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            X_batch = X[start:end].toarray()  # Convert sparse matrix to dense
            y_batch = y[start:end].values  # Ensure y is in the correct format
            yield (X_batch, y_batch)

# Train the TensorFlow Model using the Generator
model.fit(generate_batches(X_train, y_train, batch_size=32),
          steps_per_epoch=np.ceil(X_train.shape[0] / 32),
          epochs=5)

# Evaluate the TensorFlow Model using the Generator
model.evaluate(generate_batches(X_test, y_test, batch_size=32),
               steps=np.ceil(X_test.shape[0] / 32))

# Save the model and the vectorizer
model.save('toxic_comment_classifier.keras')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
