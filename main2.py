import sys, os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers


#-------------------------------------------------------------
# FUNCTION TO NORMALIZE THE PIXEL VALUES OF AN IMAGE
#-------------------------------------------------------------
def normalizeFacesData(FER_DATA):
  faces = []
  for pixel_sequence in FER_DATA['pixels']:
    temp_list = [int(s)for s in pixel_sequence.split(" ")]
    temp_list = np.reshape(temp_list, (height, width, 1))
    temp_list = temp_list/255.0
    faces.append(temp_list.astype('float32'))
  faces = np.array([np.array(face) for face in faces])
  return faces
#-------------------------------------------------------------
# HYPERPARAMETER DECLARATIONS
#-------------------------------------------------------------
classes = 7
width, height = 48,48
input_shape = (48,48,1)
batch_size = 128
epochs = 15
num_features = 64
#-------------------------------------------------------------
FER_DATA = pd.read_csv('fer2013.csv')
FER_DATA = FER_DATA.drop(['Usage'], axis = 1)
column_titles = ['pixels', 'emotion']
FER_DATA = FER_DATA.reindex(columns = column_titles)
faces = normalizeFacesData(FER_DATA)
faces = np.asarray(faces)
emotions = FER_DATA['emotion'].to_numpy()
emotions = np.expand_dims(emotions, -1)
num_classes = 7
#-------------------------------------------------------------
#KERAS MODEL 
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(num_classes, activation="relu",kernel_regularizer=l2(0.001)),
        layers.Dropout(0.50),
        layers.Dense(num_classes, activation="softmax"),
        
        layers.Dropout(0.5),
        
    ]
)

model.summary()
#-------------------------------------------------------------

#-------------------------------------------------------------
#TRAINING
#-------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size = 0.3, random_state = 1)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train,epochs=100,verbose=1,)
#-------------------------------------------------------------
'''
#-------------------------------------------------------------
#REAL-TIME VIDEO FACIAL EMOTION RECOGNITION
#-------------------------------------------------------------
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# model = load_model(MODELPATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#-------------------------------------------------------------'''