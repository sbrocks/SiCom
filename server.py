#import object_detection_api
import os
from PIL import Image
from flask import Flask, request, Response, render_template
from importlib import import_module


app = Flask(__name__)

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/')
def index():
    return Response('Tensor Flow object detection')


@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")

#import os
import numpy as np
import cv2
from keras.preprocessing import image
from camera_opencv import Camera


@app.route('/test')
def inndex():
    """Video streaming home page."""
    return render_template('inndex.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        #print(frame)
        #temp = frame
        #print(temp.decode("cp850"))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/test/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/demo')
def test():
    cap = cv2.VideoCapture(0)

    #-----------------------------
    #face expression recognizer initialization
    from keras.models import model_from_json
    model = model_from_json(open("asl_new_model.json", "r").read())
    model.load_weights('asl_new_model_weight.h5') #load weights

    signs = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')

    sign_frame = []
    confidence_frame = []
    while(cap.isOpened()):
        ret, img1 = cap.read()
        if ret == True:
            img=img1
            #print(img.shape)
            cv2.rectangle(img,(50,50),(300,300),(255,0,0),2)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=img[50:300,50:300]
            gray = cv2.resize(img, (48, 48)) #resize to 48x48

            from keras.preprocessing import image

            img_pixels = image.img_to_array(gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            
            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
            
            predictions = model.predict(img_pixels) #store probabilities of 7 expressions
            #print("Predictions: "+str(predictions))
            
            #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])
            
            sign = signs[max_index]

            sign_frame.append(sign)
            confidence_frame.append(predictions[0][max_index]*100)
            #print("Sign:"+str(sign))
            
            #write emotion text above rectangle
            cv2.putText(img1, sign, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow('img',img1)
            if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                break

        else:
            cap.release()
            cv2.destroyAllWindows()
            break


    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

    #print(sign_frame)
    #print(confidence_frame)
    res={}
    
    for i in range(1,len(sign_frame)):
        res.update({'dominant_sign_frame_'+str(i): sign_frame[i]})
        res.update({'confidence_frame_'+str(i): confidence_frame[i]})

    print(res)
    return str(res)

#cap.release()
cv2.destroyAllWindows()

@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        # finally run the image through tensor flow object detection`
        image_object = Image.open(image_file)
        objects = object_detection_api.get_objects(image_object, threshold)
        return objects

    except Exception as e:
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
	# without SSL
    app.run(debug=True, host='0.0.0.0')

	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))

