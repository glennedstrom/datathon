from random import shuffle
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid")) # will only have 2 outputs, scrambled or not scrambled


model.compile(
  optimizer='adam',
  loss="binary_crossentropy",
  metrics=['accuracy'])


test_imgs = np.load('data0123.npy')
test_imgs = np.squeeze(test_imgs, axis = 1)

test_imgs_scrambled = np.load('data3210.npy')
test_imgs_scrambled = np.squeeze(test_imgs_scrambled, axis = 1)



totalImgs = np.concatenate((test_imgs, test_imgs_scrambled), axis=0)

test_labels = np.zeros(np.shape(test_imgs)[0])
test_imgs_scrambled_labels = np.full(np.shape(test_imgs_scrambled)[0], 1)

total_labels = np.concatenate((test_labels, test_imgs_scrambled_labels), axis = 0)


print(np.shape(totalImgs))
print(np.shape(test_labels))


model.fit(
  totalImgs,
  total_labels,
  epochs = 1,
  shuffle = True,
  validation_split = 0.1
)


testInp = np.expand_dims(test_imgs_scrambled[1337],axis=0)
print(np.shape(test_imgs[4]))
print(np.shape(testInp))
print(model.predict(testInp))

