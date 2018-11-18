import object_detection_api
import os
from PIL import Image

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images' #cwh
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'F{}.jpg'.format(i)) for i in range(0, 4) ]
TEST_IMAGE_PATHS = 'test_set/A/A_test.jpg'
#for image_path in TEST_IMAGE_PATHS:
image = Image.open(TEST_IMAGE_PATHS)
response = object_detection_api.get_objects(image)
print("returned JSON: \n%s" % response)