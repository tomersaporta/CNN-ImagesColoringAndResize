
import h5py

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

import keras
import numpy as np
import os


COLOR_MODEL_PATH = r'ResizeAndColoring\models\coloringMode.h5'

RESIZE_MODEL_PATH = r'ResizeAndColoring\models\resizingModel.h5'

IMAGES_PATH= r'ResizeAndColoring\Test'

OUTPUT_PATH = 'ResizeAndColoring\Output'

def colorResize():
    # load coloring model
    modelColor = keras.models.load_model(COLOR_MODEL_PATH)
    # load resize model
    modelResize = keras.models.load_model(RESIZE_MODEL_PATH)

    color_me = []
    for filename in os.listdir(IMAGES_PATH):
        x = load_img(IMAGES_PATH + filename)
        x = rgb2lab(x)
        x = img_to_array(x)

        x = x[:,:,0] #L
        x = np.array(x,dtype=float)
        x = (x/255)
        color_me.append(x)

    color_me= np.expand_dims(color_me, axis=4)

    # Test model
    ABoutput = modelColor.predict(color_me)
    ABoutput = ABoutput*255

    Loutput = modelResize.predict(color_me)
    Loutput = Loutput*255


    # Output colorizations
    for i in range(len(color_me)):
        cur = np.zeros((96, 96, 3))
        cur[:,:,0] = Loutput[i][:,:,0]
        cur[:,:,1:] = ABoutput[i]
        cur = lab2rgb(cur)
        cur = array_to_img(cur)
        imsave(OUTPUT_PATH+str(i)+".png", cur)


if __name__ =="__main__":

    colorResize()