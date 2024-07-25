import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model

# Load the face detection model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

# Load the trained model
model = load_model('model/face_recognition_model_vgg16.h5')

# Dapatkan label kelas dari ImageDataGenerator (dari langkah sebelumnya)
class_names = ["Ucid", "Kroos", "Ronaldo", "GT", "Zul"]  # Sesuaikan dengan class_names dari langkah sebelumnya

# Function to get class name from class index
def get_className(classNo):
    return class_names[classNo]

while True:
    # Capture frame-by-frame
    success, imgOriginal = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)
    for x, y, w, h in faces:
        # Crop and preprocess the face region
        crop_img = imgOriginal[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = img.astype('float32') / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        try:
            # Make a prediction
            prediction = model.predict(img)
            classIndex = np.argmax(prediction)
            probabilityValue = np.amax(prediction)

            # Draw rectangle and put text for detected face
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOriginal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (x, y - 25), font, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
        except Exception as e:
            print("Error during prediction:", e)

    # Display the resulting frame
    cv2.imshow("Result", imgOriginal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
