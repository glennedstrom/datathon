from cgi import test
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
model.add(layers.Conv2D(32, (3, 3),activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid")) # will only have 2 outputs, scrambled or not scrambled


model.compile(
  optimizer='adam',
  loss="binary_crossentropy",
  metrics=['accuracy'])

#arrConversion = imgConvert(img_name)
#                dataList.append(arrConversion)
#            np.save('data'+ combo +'.npy', dataList)

'''unscrambled_imgs = []
for img_file in glob("unscrambled_imgs/*"):
    img = load_img(img_file, target_size=(128, 128))
    # Converts the image to a 3D numpy array (128x128x3)
    img_array = img_to_array(img)
    # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
    img_tensor = np.expand_dims(img_array, axis=0)
    unscrambled_imgs.append(img_tensor)

np.save(f'data_unscrambled.npy', unscrambled_imgs)
   ''' 


test_imgs = np.load('data0123.npy') # unscrambled imgs
# test_imgs2 = np.load('data_unscrambled.npy')
# test_imgs = np.concatenate((test_imgs, test_imgs2), axis = 0)
test_imgs = np.squeeze(test_imgs, axis = 1)


test_imgs_scrambled = np.load('data3201.npy')
test_imgs_scrambled = np.squeeze(test_imgs_scrambled, axis = 1)


totalImgs = test_imgs

test_labels = np.zeros(np.shape(test_imgs)[0])


unseen_imgs = np.load('data2031.npy')
unseen_imgs = np.squeeze(unseen_imgs, axis = 1)


totalImgs = np.concatenate((test_imgs, test_imgs_scrambled), axis=0)


test_labels = np.zeros(np.shape(test_imgs)[0])
test_imgs_scrambled_labels = np.full(np.shape(test_imgs_scrambled)[0], 1)


total_labels = np.concatenate((test_labels, test_imgs_scrambled_labels), axis = 0)


model.fit(
  totalImgs,
  total_labels,
  epochs = 1,
  shuffle = True,
  validation_split = 0.1
)



testInp = np.expand_dims(unseen_imgs[900],axis=0)

#print(np.shape(test_imgs))
#print(np.shape(totalImgs))
#print(np.shape(total_labels))
print(np.shape(test_imgs2))

#print(model.predict(testInp))

model.save("seq_model4.h5")


