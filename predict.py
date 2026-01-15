import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# Load model
model = tf.keras.models.load_model("model.h5")
CLASSES = sorted(os.listdir("C:\\Users\\omkar\\OneDrive\\Desktop\\sign_language_project\\dataset"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x,x_min), min(y,y_min)
                x_max, y_max = max(x,x_max), max(y,y_max)

            # Crop hand
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (64,64))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            prediction = model.predict(hand_img)
            class_id = np.argmax(prediction)
            sign = CLASSES[class_id]

            cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
            cv2.putText(frame, sign, (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
