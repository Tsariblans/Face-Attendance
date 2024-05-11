import cv2
import mysql.connector
import datetime
import numpy as np
from keras.models import load_model

# Load the pre-trained face detection model
face_detection_model = load_model("keras_model.h5")

# Load the labels for the Keras model
class_names = open("labels.txt", "r").readlines()

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="chwry",
    password="pass",
    database="attendance"
)

def log_face_detection(label):
    cursor = conn.cursor()
    # Get current date and time
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    # Insert face detection record into the database
    cursor.execute("INSERT INTO face_detections (date, time, label) VALUES (%s, %s, %s)", (current_date, current_time, label))
    conn.commit()
    print("Present")

def predict_label(image):
    # Preprocess the image for model input
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image.astype('float32') / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    # Perform prediction using the face detection model
    prediction = face_detection_model.predict(input_image)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    label = class_names[index].strip()  # Get the corresponding label from the class names

    return label, confidence_score

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform face detection using the model
        label, confidence_score = predict_label(frame)

        # Log face detections with predicted label
        if confidence_score > 0.5:  # Adjust confidence threshold as needed
            log_face_detection(label)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()