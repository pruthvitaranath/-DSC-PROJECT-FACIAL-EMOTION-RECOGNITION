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
epochs = 0
num_features = 64
#-------------------------------------------------------------
FER_DATA = pd.read_csv('../input/fer2013/fer2013.csv')
FER_DATA = FER_DATA.drop(['Usage'], axis = 1)
column_titles = ['pixels', 'emotion']
FER_DATA = FER_DATA.reindex(columns = column_titles)
faces = normalizeFacesData(FER_DATA)
faces = np.asarray(faces)
emotions = FER_DATA['emotion'].to_numpy()
emotions = np.expand_dims(emotions, -1)
#-------------------------------------------------------------
#KERAS MODEL 
#-------------------------------------------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                        input_shape=(48, 48, 1)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.summary()
#-------------------------------------------------------------

#-------------------------------------------------------------
#TRAINING
#-------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size = 0.3, random_state = 1)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
model.fit(x_train, y_train,epochs=100,verbose=1,)
#-------------------------------------------------------------

#-------------------------------------------------------------
#REAL-TIME VIDEO FACIAL EMOTION RECOGNITION
#-------------------------------------------------------------
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model.save('facial_1')

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
