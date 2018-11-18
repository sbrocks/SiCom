# IMPORTS

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import json
import cv2
from keras.preprocessing import image


# Model Preparation
from keras.models import model_from_json
model = model_from_json(open("asl_new_model.json", "r").read())
model.load_weights('asl_new_model_weight.h5') #load weights

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  # For colored image model: 
  # return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name="webrtcHacks TensorFlow Object Detection REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)


signs = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')

def get_objects(img, threshold=0.5):
    image_np = load_image_into_numpy_array(img)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    #print(image_np_expanded)
    #print(faces) #locations of detected faces
    sign_list = []


    #cv2.rectangle(image_np,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
    
    #detected_face = image_np[int(y):int(y+h), int(x):int(x+w)] #crop detected face
    #detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(image_np, (48, 48)) #resize to 48x48
    
    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    
    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    predictions = model._make_predict_function(image_np)
    #predictions = model.predict(img_pixels) #store probabilities of 7 expressions
    
    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
    max_index = np.argmax(predictions[0])
    
    sign = signs[max_index]
    sign_list.append(sign)

    res = {'dominant_sign': sign,
            'confidence': predictions[0][max_index]*100
    }
    
    return res


"""


    outputJson = json.dumps([ob.__dict__ for ob in output])
    return outputJson
"""

