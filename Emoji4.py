import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the emotion model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('prj.h5')  # Ensure this path is correct

# Emotion dictionary
emotion_dict = {0: 'Angry', 1: 'Sad', 2: 'Surprise', 3: 'Happy', 4: 'Neutral', 5: 'Disgusted', 6: 'Fear'}

# Set emoji paths
cur_path = os.getcwd()  # Get current working directory
emoji_dist = {
    0: os.path.join(cur_path, "emojis", "angry.png"),
    1: os.path.join(cur_path, "emojis", "sad.png"),
    2: os.path.join(cur_path, "emojis", "surprise.png"),
    3: os.path.join(cur_path, "emojis", "happy.png"),
    4: os.path.join(cur_path, "emojis", "neutral.png"),
    5: os.path.join(cur_path, "emojis", "disgust.png"),
    6: os.path.join(cur_path, "emojis", "fear.png")
}

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)  # Change to 1 if you want to test the second camera

# Load the frontal-face cascade classifier
face_cascade = cv2.CascadeClassifier("C:/Users/Divyansh/OneDrive/Desktop/face/haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # For each detected face, focus on the face area
        face = gray[y:y+h, x:x+w]
        
        # Check if the face region is not empty
        if face.size == 0:
            continue
        
        # Resize and prepare the face for prediction
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        
        # Predict emotion
        prediction = emotion_model.predict(face)
        emotion_label = np.argmax(prediction)
        emotion_emoji_path = emoji_dist[emotion_label]
        emoji_img = cv2.imread(emotion_emoji_path)

        # Resize the emoji to fit the detected face region
        emoji_img = cv2.resize(emoji_img, (w, h))
        frame[y:y+h, x:x+w] = emoji_img  # Replace face area with the emoji
        
    # Display the frame with emojis
    cv2.imshow('Emotion-Emoji', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
