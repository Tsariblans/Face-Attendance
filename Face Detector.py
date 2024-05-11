import numpy as np
import cv2
import tensorflow.keras as tf
import os
import pytesseract
import re
import mysql.connector
import datetime

# Path configurations
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LABELS_PATH = f"{DIR_PATH}/Model/labels.txt"
MODEL_PATH = f"{DIR_PATH}/Model/keras_model.h5"

# Database configurations
DB_CONFIG = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "your_database"
}

# Connect to MySQL database
conn = mysql.connector.connect(**DB_CONFIG)

def load_labels(file_path):
    with open(file_path, 'r') as labels_file:
        return [line.split(' ', 1)[1].rstrip() for line in labels_file.readlines()]

def validate_face_recognition(face_rec):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM face_info WHERE face_rec = %s", (face_rec,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None

def log_facial_entry(face_rec):
    cursor = conn.cursor()
    today = datetime.date.today()
    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    cursor.execute("SELECT last_logged_at FROM face_info WHERE face_rec = %s", (face_rec,))
    last_logged_at = cursor.fetchone()
    cooldown_period = 5  # Adjust cooldown period in seconds

    if last_logged_at:
        last_logged_at_seconds = datetime.datetime.strptime(last_logged_at[0], "%H:%M:%S")
        time_diff = datetime.datetime.now() - last_logged_at_seconds
        if time_diff.total_seconds() < cooldown_period:
            print(f"Cooldown active for student {face_rec}. Skipping entry.")
            return

    cursor.execute("INSERT INTO logs (face_rec, date, time) VALUES (%s, %s, %s)", (face_rec, today, now_str))
    cursor.execute("UPDATE face_info SET last_logged_at = %s WHERE face_rec = %s", (now_str, face_rec))
    conn.commit()
    print(f"Student {face_rec} logged in successfully!")

def main():
    classes = load_labels(LABELS_PATH)
    model = tf.models.load_model(MODEL_PATH, compile=False)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing video frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 0:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image_array = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        predictions = model.predict(data)
        conf_threshold = 90

        for i, confidence in enumerate(predictions[0]):
            if confidence * 100 >= conf_threshold:
                face_info = validate_face_recognition(classes[i])
                if face_info:
                    log_facial_entry(face_info)
                else:
                    print("Not a student!")

        cv2.imshow("Capturing", frame)
        if cv2.waitKey(10) == 27:  # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
