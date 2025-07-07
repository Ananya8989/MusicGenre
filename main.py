
#%%
import os, sys
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential,Model, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import scipy
import sys, os, pickle
import librosa
from sklearn.model_selection import train_test_split
import IPython
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
from IPython.display import display
import tensorflow
from tensorflow.python.keras.layers import Input, Dense
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

def main():
 df = pd.read_csv('C:/users/anany/Downloads/archive/Data/features_3_sec.csv')
 df.head()
 df.shape
 df.dtypes
 df = df.drop(labels='filename',axis=1)
'''
dataR, samplingRate = librosa.load('C:/Users/anany/Downloads/archive/Data/genres_original/rock/rock.00066.wav')
print(type(dataR),type(samplingRate))
librosa.load('C:/Users/anany/Downloads/archive/Data/genres_original/rock/rock.00066.wav',sr=45600)
IPython.display.Audio(dataR,rate=samplingRate)

dataC, samplingRate = librosa.load('C:/Users/anany/Downloads/archive/Data/genres_original/classical/classical.00066.wav')
print(type(dataC),type(samplingRate))
librosa.load('C:/Users/anany/Downloads/archive/Data/genres_original/classical/classical.00066.wav',sr=45600)
IPython.display.Audio(dataC,rate=samplingRate)

#audio paths/labels 
count = 0
base = "C:/Users/anany/"
nameDir = base+ "Downloads/archive/Data/genres_original"
paths = []
labels = []
print(nameDir)
for rt, dir, fileCollect in os.walk(nameDir, topdown = False):
    for fileName in fileCollect:

        if fileName.find('.wav') != -1:

            paths.append(os.path.join(rt, fileName))
            fileName = fileName.split('.', 1)
            fileName = fileName[0]
            labels.append(fileName)
            print(len(labels))
paths = np.array(paths)
labels = np.array(labels)

#Feature Extraction
SpecArr = np.empty([1000, 1025, 1293])
MelArr = np.empty([1000, 128, 1293])
MfccArr = np.empty([1000, 10, 1293])

count = 0
# Creating a list for the corrupt indices
corrupt = []
for i in tqdm(range(len(paths))):
    try:
        path = paths[i]
        y1, sr = librosa.load(path)
        X = librosa.stft(y1)
        dbx = librosa.amplitude_to_db(abs(X))
        SpecArr[i] = dbx

        M = librosa.feature.melspectrogram(y=y1)
        M_db = librosa.power_to_db(M)
        MelArr[i] = M_db

        m = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc= 10)
        MfccArr[i] = m
    
    except Exception as e:
       SpecArr[i] = 0
       corrupt.append(i)
       print(i)

SpecArr = np.delete(SpecArr,corrupt,0)
MelArr = np.delete(MelArr,corrupt,0)
MfccArr = np.delete(MfccArr,corrupt,0)
print('done 0')
SpecArr = SpecArr.astype(np.float32)
MelArr = MelArr.astype(np.float32)
MfccArr = MfccArr.astype(np.float32)
print('done 1')
labels = np.delete(labels, corrupt)
labels[labels == 'blues'] = 0
labels[labels == 'classical'] = 1
labels[labels == 'country'] = 2
labels[labels == 'disco'] = 3
labels[labels == 'hiphop'] = 4
labels[labels == 'jazz'] = 5
labels[labels == 'metal'] = 6
labels[labels == 'pop'] = 7
labels[labels == 'reggae'] = 8
labels[labels == 'rock'] = 9
labels = [int(i) for i in labels]
labels = np.array(labels)
    
y = tensorflow.keras.utils.to_categorical(labels,num_classes = 10, dtype ='float32')
np.savez_compressed(base+"/MusicFeatures.npz", spec= SpecArr, mel= MelArr, mfcc= MfccArr, target=y) 
print("done 2")

f = np.load(base+"/MusicFeatures.npz")
S = f['spec']
mfcc = f['mfcc']
mel = f['mel']
y = f['target']
STrain, STest, mfccTrain, mfccTest, melTrain, melTest, yTrain, yTest = train_test_split(S, mfcc, mel, y, test_size= 0.2)
print("done 3")
max1 = np.amax(STrain)
STrain = STrain/np.amax(max1)
STest = STest/np.amax(max1)

STrain = STrain.astype(np.float32)
STest = STest.astype(np.float32)

N, r, c = STrain.shape
STrain = STrain.reshape((N, r, c, 1))

N, r, c = STest.shape
STest = STest.reshape((N, r, c, 1))
print('done 4')

#Reshape and standardize data
mfccTrain2 = np.empty([mfccTrain.shape[0], 120,600])
mfccTest2 = np.empty([mfccTest.shape[0],120,600])

for i in range(mfccTrain.shape[0]) :
    current = mfccTrain[i]
    current = cv2.resize(current, (600, 120))
    mfccTrain2[i] = current

mfccTrain = mfccTrain2

for i in range(mfccTest.shape[0]) :
  current = mfccTest[i]
  current = cv2.resize(current, (600, 120))
  mfccTest2[i] = current

mfccTest = mfccTest2

mfccTrain = mfccTrain.astype(np.float32)
mfccTest = mfccTest.astype(np.float32)

N, r, c = mfccTrain.shape
mfccTrain = mfccTrain.reshape([N, r, c, 1])

N, r, c = mfccTest.shape
mfccTest = mfccTest.reshape([N, r, c, 1])

mean = np.mean(mfccTrain)
sd = np.std(mfccTrain)

mfccTrain = (mfccTrain - mean)/sd
mfccTest = (mfccTest - mean)/sd

print('done 5')

max = np.amax(melTrain)
melTrain = melTrain/(np.amax(max))
melTest = melTest/(np.amax(max))

melTrain = melTrain.astype(np.float32)
melTest = melTest.astype(np.float32)

N, r, c = melTrain.shape
melTrain = melTrain.reshape([N, r, c, 1])

N, r, c = melTest.shape
melTest = melTest.reshape([N, r, c, 1])

np.savez_compressed(base+"/new_spectrogram_train_test.npz", S_train= STrain, S_test= STest, y_train = yTrain, y_test= yTest)

np.savez_compressed(base+"/new_mfcc_train_test.npz", mfcc_train= mfccTrain, mfcc_test= mfccTest, y_train = yTrain, y_test= yTest)

np.savez_compressed(base+"/new_mel_train_test.npz", mel_train= melTrain, mel_test= melTest, y_train = yTrain, y_test= yTest)'''
base = "C:/Users/anany/"

sf = np.load(base + "/new_spectrogram_train_test.npz")
spec_file = np.load(base+"/new_spectrogram_train_test.npz")

# Model 1 for Spectrogram
STrain = spec_file['S_train']
STest = spec_file['S_test']
yTrain = spec_file['y_train']
yTest = spec_file['y_test']

model = Sequential()

model.add(Conv2D(8,(3,3), activation = 'relu', input_shape = STrain[0].shape, padding = 'same'))
model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
model.add(MaxPooling2D((4,4), padding= 'same'))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')

model.summary()

#checkpoint = ModelCheckpoint(base+"/models/new_spec_model_spectrogram1_{epoch:03d}.h5", period= 5)

#model.fit(STrain, yTrain, epochs= 100, callbacks= [checkpoint], batch_size= 32, verbose= 1)
model.fit(STrain, yTrain, epochs= 20, batch_size= 32, verbose= 1)
model.save(base + "/models/new_spec_model_spectrogram1.h5", save_format='tf')

model = load_model(base + "/models/new_model_spectrogram1.h5")

# Training Accuracy
yPredicted = model.predict(STrain)
yPredicted = np.argmax(yPredicted, axis= -1)
yActual = np.argmax(yTrain, axis= -1)

correct = len(yPredicted) - np.count_nonzero(yPredicted - yActual)
acc = correct/ len(yPredicted)
acc = np.round(acc, 4) * 100

print("Train Accuracy: ", correct, "/", len(yPredicted), " = ", acc, "%")


# %%
