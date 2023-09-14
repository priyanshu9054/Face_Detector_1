import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face_LBPHFaceRecognizer.create()

path = "dataset"

def getImageID(path):
    imagePathList = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagePathList:
        faceImage = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImage)
        Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Fixed the typo and conversion issue
        faces.append(faceNP)
        ids.append(Id)
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)

    return ids, faces

IDs, faces = getImageID(path)
recognizer.train(faces, np.array(IDs))
recognizer.save("Face1.yaml")  # Changed write to save
cv2.destroyAllWindows()
print("Training completed..............")
