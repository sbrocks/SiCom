# IMPORTS

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt ### CWH
from PIL import Image


# What model to download.
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model.json", "r").read())
model.load_weights('facial_expression_model_weight.h5') #load weights


PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images' #cwh
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]


for image_path in TEST_IMAGE_PATHS:
	image = Image.open(image_path)
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
