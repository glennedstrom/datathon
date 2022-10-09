from random import shuffle
from glob import glob
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
#import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = not scrambled, 1 = scrambled
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


test_imgs = np.load('data0123.npy') # unscrambled imgs
test_imgs = np.squeeze(test_imgs, axis = 1)

test_imgs_scrambled = np.load('data3210.npy')
test_imgs_scrambled = np.squeeze(test_imgs_scrambled, axis = 1)

totalImgs = test_imgs

test_labels = np.zeros(np.shape(test_imgs)[0])


unseen_imgs = np.load('data2031.npy')
unseen_imgs = np.squeeze(unseen_imgs, axis = 1)


totalImgs = np.concatenate((test_imgs, test_imgs_scrambled), axis=0)


test_labels = np.zeros(np.shape(test_imgs)[0])
test_imgs_scrambled_labels = np.full(np.shape(test_imgs_scrambled)[0], 1)


total_labels = np.concatenate((test_labels, test_imgs_scrambled_labels), axis = 0)
total_labels = np.concatenate((total_labels, test_imgs_scrambled_labels), axis = 0)


print(np.shape(totalImgs))
print(np.shape(test_labels))


"""model.fit(
  totalImgs,
  total_labels,
  epochs = 1,
  shuffle = True,
  validation_split = 0.1
)"""


testInp = np.expand_dims(unseen_imgs[900],axis=0)
print(np.shape(test_imgs[4]))
print(np.shape(testInp))

print(np.shape(totalImgs))
print(np.shape(total_labels))
#print(model.predict(testInp))

model.save("seq_model2.h5")

model.save('saved_models/seqModel.h5')

