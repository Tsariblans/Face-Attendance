from flask import Flask, render_template, Response
import cv2
import numpy as np
import datetime
from keras.models import load_model
import mysql.connector

app = Flask(__name__)

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
    if label != "Unknown":
        cursor = conn.cursor()
        # Get current date and time
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
        # Insert face detection record into the database
        cursor.execute("INSERT INTO face_detections (date, time, label) VALUES (%s, %s, %s)", (current_date, current_time, label))
        conn.commit()
        return "Present"
    else:
        return "Unknown face detected. Please try again."



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

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



from flask import request

@app.route('/detect_face')
def detect_face():
    label, confidence_score = predict_label(cv2.VideoCapture(0).read()[1])
    result = log_face_detection(label)
    return result



@app.route('/try_again')
def try_again():
    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)
