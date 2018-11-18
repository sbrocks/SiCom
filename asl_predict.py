import numpy as np
import cv2
from keras.preprocessing import image

cap = cv2.VideoCapture(0)

""" We use cv2.VideoWriter() to save the video """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_shrofile.avi',fourcc,20.0,(640,480))



#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("asl_new_model.json", "r").read())
model.load_weights('asl_new_model_weight.h5') #load weights

signs = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')

cnt =0 
while(cap.isOpened()):
	ret, img1 = cap.read()
	if ret == True:
		img=img1
		#print(img.shape)
		cv2.rectangle(img,(150,150),(300,300),(255,0,0),2)
		#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img=img[150:300,150:300]
		gray = cv2.resize(img, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(gray)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		#print("Predictions: "+str(predictions))
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		sign = signs[max_index]
		#print("Sign:"+str(sign))
		
		#write emotion text above rectangle
		cv2.putText(img1, sign, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

		from gtts import gTTS
		import os
		tts = gTTS(text='Good morning', lang='en')
		tts.save("good"+str(count)+".mp3")
		os.system("mpg321 good+str(count)+.mp3")

		#out.write(img)
		cv2.imshow('img',img1)
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	else:
		break


# Release the webcam
cap.release()
out.release()
cv2.destroyAllWindows()
