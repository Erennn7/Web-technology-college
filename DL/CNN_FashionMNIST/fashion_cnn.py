# ==========================================================
# CNN Model for Fashion Image Classification using Fashion-MNIST
# ==========================================================

# Step 1: Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------
# Step 2: Load Dataset
# ----------------------------------------------------------

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

# ----------------------------------------------------------
# Step 3: Display Sample Images
# ----------------------------------------------------------

class_names = [
    'T-shirt/Top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
]

plt.figure(figsize=(8, 8))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 4: Preprocessing
# ----------------------------------------------------------

# Normalize images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape images
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ----------------------------------------------------------
# Step 5: Build CNN Model
# ----------------------------------------------------------

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        10,
        activation='softmax'
    )
)

# ----------------------------------------------------------
# Step 6: Compile Model
# ----------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print("\nModel Summary:\n")
model.summary()

# ----------------------------------------------------------
# Step 7: Train Model
# ----------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# ----------------------------------------------------------
# Step 8: Evaluate Model
# ----------------------------------------------------------

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Loss     :", loss)
print("Test Accuracy :", accuracy)

# ----------------------------------------------------------
# Step 9: Plot Accuracy
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# Step 10: Plot Loss
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

# ----------------------------------------------------------
# Step 11: Predict Test Images
# ----------------------------------------------------------

predictions = model.predict(X_test)

predicted = np.argmax(predictions, axis=1)
actual = np.argmax(y_test, axis=1)

# ----------------------------------------------------------
# Step 12: Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(actual, predicted)

print("\nConfusion Matrix:\n")
print(cm)

# ----------------------------------------------------------
# Step 13: Classification Report
# ----------------------------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        actual,
        predicted,
        target_names=class_names
    )
)

# ----------------------------------------------------------
# Step 14: Display Sample Predictions
# ----------------------------------------------------------

plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(
        f"Pred: {class_names[predicted[i]]}\nActual: {class_names[actual[i]]}"
    )
    plt.axis('off')

plt.tight_layout()
plt.show()