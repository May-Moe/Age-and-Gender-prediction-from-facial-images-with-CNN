import numpy as np
import cv2
from keras.models import load_model
from collections import deque
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
import argparse

# Argument parser to select model
parser = argparse.ArgumentParser(description='Age and Gender Prediction using Webcam')
parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
args = parser.parse_args()

# Load the model
print(f"Loading the model from {args.model}...") 
model = load_model(args.model)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")

# Queues to store the last N age and gender predictions
age_predictions = deque(maxlen=10)
gender_predictions = deque(maxlen=10)

def get_stable_prediction(predictions):
    """Helper function to get the most stable prediction"""
    return max(set(predictions), key=predictions.count)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if w == 0 or h == 0:
                continue
            img = gray[y-50:y+40+h, x-10:x+10+w]
            if img.size == 0:
                continue
            
            try:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (200, 200))

                # Check input shape before prediction
                print(f"Input shape: {img.shape}")
                
                # Ensure the image is in the correct shape
                img = np.array(img).reshape(-1, 200, 200, 3)
                
                # Predict
                predict = model.predict(img)
                
                age_predictions.append(predict[0][0])
                gender_predictions.append(np.argmax(predict[1]))

                if len(age_predictions) >= 10:
                    stable_age = np.mean(age_predictions)
                    stable_gender = get_stable_prediction(gender_predictions)

                    if stable_gender == 0:
                        gend = 'Man'
                        col = (255, 0, 0)
                    else:
                        gend = 'Woman'
                        col = (203, 12, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 4)
                    cv2.putText(frame, "Age : " + str(int(stable_age)) + " - " + str(gend), (x, y), cv2.FONT_HERSHEY_SIMPLEX, w*0.005, col, 4)
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

        cv2.imshow('Webcam - Age and Gender Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
