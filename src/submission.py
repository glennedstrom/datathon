# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
from xmlrpc.client import MAXINT
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import itertools

# Import helper functions from utils.py
import utils

def colorDiff(rgb1, rgb2):
    s = 0
    for i in range(0,3):
        s += (rgb1[i]-rgb2[i])**2
    return s**0.5

def compare(quadrents):  # quadrents[quad][rows][rgb]
    #quad 0 and quad 1 loop through 63 and 0
    sum = 0
    for i in range(0,63):#vertical middle
        sum += colorDiff(quadrents[0][i][0][63], quadrents[1][i][0][0])
        sum += colorDiff(quadrents[2][i][0][63], quadrents[3][i][0][0])
    for i in range(0,63):#horizontal stripe
        sum += colorDiff(quadrents[0][63][0][i], quadrents[2][0][0][i])
        sum += colorDiff(quadrents[1][63][0][i], quadrents[3][0][0][i])
    return sum


    

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        self.model = load_model('example_model.h5')

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path: 
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`
        
        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """

        # Load the image
        img = load_img(f'{img_path}', target_size=(128, 128))

        # Converts the image to a 3D numpy array (128x128x3)
        img_array = img_to_array(img)

        # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
        img_tensor = np.expand_dims(img_array, axis=0)

        # Preform a prediction on this image using a pre-trained model (you should make your own model :))
        prediction = self.model.predict(img_tensor, verbose=False)

        # The example model was trained to return the percent chance that the input image is scrambled using 
        # each one of the 24 possible permutations for a 2x2 puzzle
        #quadrents = [[],[],[],[]] # [left, right, up, down]
        quadrents = []
        
        for row in [0,1]:
            for col in [0,1]:
                l = col*64
                u = row*64

                #quadrents.append(img_array[l:l+64][u:u+64])
                quadrents.append([[i[u:u+64]] for i in img_array[l:l+64]])
                print([[i[u:u+64]] for i in img_array[l:l+64]])
        
        min = MAXINT
        for i, combo in enumerate(itertools.permutations([0,1,2,3])):
            print(i, combo)
        for i, combo in enumerate(itertools.permutations(quadrents)):
            c = compare(combo)
            print(i, c)
            if c < min:
                min = c

        quadrents = []
        
        for row in [0,1]:
            for col in [0,1]:
                l = col*64
                u = row*64
                quadrents.append([l,u])
        


        print(quadrents)
        sums = []
        # should loop through one of each pair of quadrents
        for i in range(1,4):#quadrent i
            horizontalMin = (-1,-1)# ((quadrent1, quadrent2), value)
            verticalMin = (-1,-1)
            for j in range(0,i):#quadrent j
                verticalSum = 0
                horizontalSum = 0
                #loop through edges
                for k in range(0,64):
                    #left and right edges
                    verticalSum += colorDiff(img_array[quadrents[i][1] + k][quadrents[i][0]], img_array[quadrents[j][1] + k][quadrents[j][0]])
                    horizontalSum += colorDiff(img_array[quadrents[i][1]][quadrents[i][0] + k], img_array[quadrents[j][1]][quadrents[j][0] + k])
                if horizontalMin[1] == -1 or horizontalSum < horizontalMin[1]:
                    horizontalMin = ((i,j), horizontalSum)
                if verticalMin[1] == -1 or verticalSum < verticalMin[1]:
                    verticalMin = ((i,j), verticalSum)
            print(verticalMin[0], horizontalMin[0],verticalMin[1], horizontalMin[1])
        
                


                    


                #check edges



        combs = [''.join(str(x) for x in comb) for comb in list(permutations(range(0, 4)))]

        # Return the combination that the example model thinks is the solution to this puzzle
        # Example return value: `3120`
        return combs[np.argmax(prediction)]

# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    for img_name in glob('example_images/*'):
        # Open an example image using the PIL library
        example_image = Image.open(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        # Example images are all shuffled in the "3120" order
        print(prediction)

        # Visualize the image
        pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # Example images are all shuffled in the "3120" order
        final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        final_image.show()
        