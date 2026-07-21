import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===========================
# Step 1: Load Dataset
# ===========================

data = pd.read_csv("Tweets.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# ===========================
# Step 2: Select Columns
# ===========================

tweets = data["text"]
labels = data["airline_sentiment"]

# ===========================
# Step 3: Clean Tweets
# ===========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

tweets = tweets.apply(clean_text)

# ===========================
# Step 4: Encode Labels
# ===========================

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

print("\nClasses:", encoder.classes_)

y = to_categorical(y)

# ===========================
# Step 5: Tokenization
# ===========================

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(tweets)

sequences = tokenizer.texts_to_sequences(tweets)

# ===========================
# Step 6: Padding
# ===========================

max_length = 50

X = pad_sequences(
    sequences,
    maxlen=max_length
)

print("\nShape after Padding:", X.shape)

# ===========================
# Step 7: Train-Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ===========================
# Step 8: Build LSTM Model
# ===========================

model = Sequential()

model.add(
    Embedding(
        input_dim=max_words,
        output_dim=128
    )
)

model.add(
    LSTM(
        128,
        return_sequences=False
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        3,
        activation="softmax"
    )
)

# ===========================
# Step 9: Compile Model
# ===========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary")
model.summary()

# ===========================
# Step 10: Train Model
# ===========================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.20
)

# ===========================
# Step 11: Evaluate Model
# ===========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

# ===========================
# Step 12: Predictions
# ===========================

predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis=1)
actual = np.argmax(y_test, axis=1)

# ===========================
# Step 13: Confusion Matrix
# ===========================

cm = confusion_matrix(
    actual,
    predictions
)

print("\nConfusion Matrix\n")
print(cm)

# ===========================
# Step 14: Classification Report
# ===========================

print("\nClassification Report\n")

print(
    classification_report(
        actual,
        predictions,
        target_names=encoder.classes_
    )
)

# ===========================
# Step 15: Accuracy Graph
# ===========================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])

plt.show()

# ===========================
# Step 16: Loss Graph
# ===========================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])

plt.show()

# ===========================
# Step 17: Test with New Tweet
# ===========================

sample = [
    "The flight was comfortable and the staff were very friendly."
]

sample_seq = tokenizer.texts_to_sequences(sample)

sample_pad = pad_sequences(
    sample_seq,
    maxlen=max_length
)

prediction = model.predict(sample_pad)

predicted_label = encoder.inverse_transform(
    [np.argmax(prediction)]
)

print("\nSample Tweet:")
print(sample[0])

print("\nPredicted Sentiment:")
print(predicted_label[0])