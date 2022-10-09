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
model.add(layers.Dense(2)) # will only have 2 outputs, scrambled or not scrambled

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


test_imgs = np.load('data0123.npy')
print(test_imgs[0])


print(model.predict(test_imgs[0]))

