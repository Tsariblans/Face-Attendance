from flask import Flask, render_template, Response
import cv2
import numpy as np
import datetime
from keras.models import load_model
import mysql.connector

app = Flask(__name__)


face_detection_model = load_model("keras_model.h5")


class_names = open("labels.txt", "r").readlines()


conn = mysql.connector.connect(
    host="localhost",
    user="chwry",
    password="pass",
    database="attendance"
)

def log_face_detection(label):
    if label != "Unknown":
        cursor = conn.cursor()
        
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
       
        cursor.execute("INSERT INTO face_detections (date, time, label) VALUES (%s, %s, %s)", (current_date, current_time, label))
        conn.commit()
        return "Present"
    else:
        return "Unknown face detected. Please try again."



def predict_label(image):
    
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image.astype('float32') / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    
    prediction = face_detection_model.predict(input_image)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    label = class_names[index].strip()  

    return label, confidence_score

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()  
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
    cap.release()

@app.route('/')
def index():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM face_detections")
    data = cursor.fetchall()
    return render_template('index.html', data=data)

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
