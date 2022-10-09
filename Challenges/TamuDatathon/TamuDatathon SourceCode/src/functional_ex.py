from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import os

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid")) # will only have 2 outputs, scrambled or not scrambled


model.compile(
  optimizer='adam',
  loss="binary_crossentropy",
  metrics=['accuracy'])


test_imgs = np.load('data0123.npy')
test_imgs = np.squeeze(test_imgs, axis = 1)
test_labels = np.zeros(np.shape(test_imgs)[0])

test_imgs_scrambled = np.load('data3210.npy')
test_imgs_scrambled = np.squeeze(test_imgs_scrambled, axis = 1)
test_labels_scrambled = np.ones(np.shape(test_imgs)[0])

totalImgs = np.concatenate((test_imgs, test_imgs_scrambled), axis=0)
totalLabels = np.concatenate((test_labels, test_labels_scrambled), axis=0)

print(np.shape(totalImgs))
print(np.shape(totalLabels))


model.fit(
  totalImgs,
  totalLabels,
  epochs = 1
)

for i in range(10):
    testInp = np.expand_dims(test_imgs[i],axis=0)
    print(model.predict(testInp))


print(model.summary())

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.listdir(checkpoint_dir)

loss, acc = model.evaluate(totalImgs, totalLabels, verbose=2)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))

