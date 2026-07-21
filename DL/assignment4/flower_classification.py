import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    Rescaling
)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ===========================
# Step 1: Load Dataset
# ===========================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers",
    validation_split=0.20,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers",
    validation_split=0.20,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# ===========================
# Step 2: Display Class Names
# ===========================

class_names = train_dataset.class_names

print("\nFlower Classes:")
print(class_names)

# ===========================
# Step 3: Visualize Images
# ===========================

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

# ===========================
# Step 4: Improve Performance
# ===========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ===========================
# Step 5: Build CNN Model
# ===========================

model = Sequential([

    Rescaling(1./255, input_shape=(180, 180, 3)),

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        5,
        activation="softmax"
    )

])

# ===========================
# Step 6: Compile Model
# ===========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary\n")

model.summary()

# ===========================
# Step 7: Train Model
# ===========================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20
)

# ===========================
# Step 8: Evaluate Model
# ===========================

loss, accuracy = model.evaluate(validation_dataset)

print("\nValidation Accuracy:", accuracy)

# ===========================
# Step 9: Accuracy Graph
# ===========================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])

plt.show()

# ===========================
# Step 10: Loss Graph
# ===========================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])

plt.show()

# ===========================
# Step 11: Predict New Flower
# ===========================

image_path = "rose.jpg"

try:

    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(180, 180)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction[0])

    confidence = np.max(tf.nn.softmax(prediction[0])) * 100

    print("\nPredicted Flower:")
    print(class_names[predicted_class])

    print("Confidence: {:.2f}%".format(confidence))

except FileNotFoundError:

    print("\nrose.jpg not found. Skipping prediction step.")

# ===========================
# Step 12: Confusion Matrix
# ===========================

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    y_pred.extend(predicted)

    y_true.extend(labels.numpy())

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix\n")

print(cm)

# ===========================
# Step 13: Classification Report
# ===========================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)