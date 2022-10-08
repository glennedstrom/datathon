import numpy as np
from glob import glob
from tensorflow.keras.utils import load_img, img_to_array
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