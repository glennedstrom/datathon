from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import tensorflow as tf
import random

img = load_img('210_128.jpg', target_size=(128, 128))
# Converts the image to a 3D numpy array (128x128x3)
img_array = img_to_array(img)
# Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
img_tensor = np.expand_dims(img_array, axis=0)



model = load_model('seq_model1.h5')

unseen_imgs = np.load('data2013.npy')
unseen_imgs = np.squeeze(unseen_imgs, axis = 1)

#img_tensor
testInp = np.expand_dims(unseen_imgs[random.randint(0, 2000)],axis=0)
print(model.predict(testInp))