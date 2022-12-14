import numpy as np
from glob import glob
from tensorflow.keras.utils import load_img, img_to_array
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def imgConvert(img_path):
    # Load the image
    img = load_img(f'{img_path}', target_size=(128, 128))
    # Converts the image to a 3D numpy array (128x128x3)
    img_array = img_to_array(img)
    # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

if __name__ == '__main__':
    comboList = []
    for i in range(4):
        for j in range(4):
            if (i!=j):
                for k in range(4):
                    if (i!=k and j!=k):
                        for l in range(4):
                            if (i!=l and j!=l and k!=l):
                                comboList.append(str(i)+str(j)+str(k)+str(l))
    singleFile = True
    if (singleFile):
        print(comboList)
        cnt = 0
        dataList = []
        for combo in comboList:
            print("Making npyFile for : " + combo)
            globby = glob('TamuDatathonTrainingData/train/' + combo + '/*')
            for img_name in range(1,len(globby)):
                cnt+=1
                if (cnt%100 == 0):
                    print('[' + 'X'*int(cnt*20/len(globby)/23) + 'O'*int((len(globby)*23-cnt)*20/len(globby)/23) + ']')
                arrConversion = imgConvert(globby[img_name])
                dataList.append(arrConversion)
        np.save('data'+ "Shuffled" +'.npy', dataList)
        print("Completed folder : " + "data Shuffled") 
    else:
        print(comboList)
        for combo in comboList:
            print("Making npyFile for : " + combo)
            dataList = []
            cnt = 0
            globby = glob('TamuDatathonTrainingData/train/' + combo + '/*')
            for img_name in globby:
                cnt+=1
                if (cnt%100 == 0):
                    print('[' + 'X'*int(cnt*20/len(globby)) + 'O'*int((len(globby)-cnt)*20/len(globby)) + ']')
                arrConversion = imgConvert(img_name)
                dataList.append(arrConversion)
            np.save('data'+ combo +'.npy', dataList)
            print("Completed folder : " + combo) 