import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from tensorflow.keras.utils import load_img, img_to_array
import utils
def imgConvert(img_path):
    # Load the image
    img = load_img(f'{img_path}', target_size=(128, 128))
    # Converts the image to a 3D numpy array (128x128x3)
    img_array = img_to_array(img)
    # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

if __name__ == '__main__':
    comboList = ['0123', '0132', '0213', '0231', '0312', '0321', '1023', '1032', '1203', '1230', '1302', '1320', '2013', '2031', '2103', '2130', '2301', '2310', '3012', '3021', '3102', '3120', '3201', '3210']
    temp = 2
    for combo in comboList:
        if (temp == 3):
            break
        dataList = []
        cnt = 0
        globby = glob('TamuDatathonTrainingData/train/' + combo + '/*')
        for img_name in globby:
            cnt+=1
            if (cnt%100):
                print("["+"X"*cnt*20/len(globby) + "O"*(len(globby)-cnt)*20/len(globby) + "]")
            arrConversion = imgConvert(img_name)
            dataList.append(arrConversion)
        np.save('data'+ combo +'.npy', dataList)
        print("Completed folder : " + combo)
        temp+=1