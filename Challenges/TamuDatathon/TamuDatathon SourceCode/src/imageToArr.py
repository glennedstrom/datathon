import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from tensorflow.keras.utils import load_img, img_to_array
import utils
def make_prediction(self, img_path):
    # Load the image
    img = load_img(f'{img_path}', target_size=(128, 128))
    
    # Converts the image to a 3D numpy array (128x128x3)
    img_array = img_to_array(img)
    
    # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

if __name__ == '__main__':
