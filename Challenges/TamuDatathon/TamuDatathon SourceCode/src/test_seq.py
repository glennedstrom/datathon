from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import tensorflow as tf
import random

img = load_img('307_128.jpg', target_size=(128, 128))
# Converts the image to a 3D numpy array (128x128x3)
img_array = img_to_array(img)
# Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
img_tensor = np.expand_dims(img_array, axis=0)



model = load_model('seq_model3.h5')

unseen_scram_imgs = np.load('data2310.npy') # dont use 3201
unseen_scram_imgs = np.squeeze(unseen_scram_imgs, axis = 1)

unseen_unscram_imgs = np.load('data_unscrambled.npy')
unseen_unscram_imgs = np.squeeze(unseen_unscram_imgs, axis = 1)

#img_tensor

scram_count = 0
for i in range(100):
    testInp = np.expand_dims(unseen_scram_imgs[random.randint(0, 2000)],axis=0)
    pred = model.predict(testInp)
    if pred[0][0] > .66:
        scram_count += 1 
    print(pred)


print("===============================================")
unscram_count = 0
for i in range(100):
    testInp = np.expand_dims(unseen_unscram_imgs[random.randint(0, 1000)],axis=0)
    pred = model.predict(testInp)
    if pred[0][0] < .33:
        unscram_count += 1 
    print(pred)

print(scram_count, "/100 scrambled")
print(unscram_count, "/100 unscrambled")
