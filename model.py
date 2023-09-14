import cv2
import numpy as np

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read("Trainer.yml")

name_list = ["", "Priyanshu"]  # You can update this list with actual names

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Changed COLOR_RGB2GRAY to COLOR_BGR2GRAY

    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf > 70:  # Lowered the confidence threshold for better recognition
            name = name_list[serial]
        else:
            name = "Unknown"

        cv2.putText(frame, name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("Face recognition done.")
