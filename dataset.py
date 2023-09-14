import cv2
import numpy as np

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get user ID as input
id = input("Enter your ID: ")

# Initialize a counter for captured images
count = 0

while True:
    # Read a frame from the video
    ret, frame = video.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        count += 1
        # Save the captured face with a unique filename
        cv2.imwrite('dataset/User.' + str(id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    
    # Show the frame with detected faces
    cv2.imshow("Face Detection", frame)
    
    # Wait for a key press and break the loop if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q') or count > 500:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

print("Dataset collection is done.")
