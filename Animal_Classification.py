# Multi Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='0'

base_dir = 'C:\\Users\\USER\\Downloads'
img_dir = 'C:\\Users\\USER\\Downloads\\animal_classification\\raw-img'
len(os.listdir(img_dir)) # category 수


data_generator = ImageDataGenerator(rescale = 1./255)
data = data_generator.flow_from_directory(img_dir, target_size=(300,300), batch_size=26179, class_mode='categorical')
x_data, y_data = data.next()
print(x_data[1000].shape)
plt.imshow(x_data[1000])
plt.show()
y_data[1000] #0:butterfly, 1:cat, 2:chicken, 3:cow, 4: dog, 5: elephant, 6:horse, 7:sheep, 8:spider, 9:squarrel

# 데이터가 잘 로드되었나 확인
N, width, height, channels = x_data.shape
n, classes = y_data.shape
print('x_shape(이미지 개수: {}, 가로: {}, 세로: {}, 채널 수: {}), \ny_shape(라벨 수:{})'.format(N, width, height, channels, classes))

# split data into train & test
with tf.device('/device:GPU:0'): 
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=1)

# Training Model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(300,300,3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.9), metrics=['acc'])
es = callbacks(monitor='val_loss', patience=10)
learn = model.fit(x_train, y_train, batch_size=160, epochs=100, callbacks=[es], validation_split=0.2)

# Evaluate Model
y_pred = model.predict(x_test)
y_true = np.argmax(y_test)
y_pred = np.argmax(y_pred)
cf_matrix = confusion_matrix(y_true, y_pred, labels=range(0,classes))

plt.subplot(1,2,1)
plt.figure(figsize=(7,7))
