import cv2
import mysql.connector
import datetime
import numpy as np
from keras.models import load_model


face_detection_model = load_model("keras_model.h5")


class_names = open("labels.txt", "r").readlines()

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="chwry", #ilisdig username ninyo
    password="pass", #password
    database="attendance" #tapos database nga inyong gibuhat
)

def log_face_detection(label):
    cursor = conn.cursor()
   
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
   
    cursor.execute("INSERT INTO face_detections (date, time, label) VALUES (%s, %s, %s)", (current_date, current_time, label))
    conn.commit()
    print("Present")

def predict_label(image):
    
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image.astype('float32') / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

   
    prediction = face_detection_model.predict(input_image)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    label = class_names[index].strip() 

    return label, confidence_score

def main():
  
    cap = cv2.VideoCapture(0)

    while True: # patabang sad ko diri dapit diko kabalo kung paano atong pag scan
       
        ret, frame = cap.read()

        
        label, confidence_score = predict_label(frame)

       
        if confidence_score > 0.5: 
            log_face_detection(label)

    
        cv2.imshow('Face Detection', frame)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
