import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# DATASET PATH
DATASET_PATH = "C:\\Users\\omkar\\OneDrive\\Desktop\\sign_language_project\\dataset"
IMG_SIZE = 64
CLASSES = sorted(os.listdir(DATASET_PATH))

X = []
y = []

# Load images
for label, folder in enumerate(CLASSES):
    folder_path = os.path.join(DATASET_PATH, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN MODEL
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# SAVE MODEL
model.save("model.h5")
print("✅ Model trained and saved as model.h5")
