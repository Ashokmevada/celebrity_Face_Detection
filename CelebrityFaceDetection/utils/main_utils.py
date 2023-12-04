import base64
import cv2 as cv
import numpy as np

def decodeImage(imgstring):
    decoded_image = base64.b64decode(imgstring)
    np_arr = np.frombuffer(decoded_image , np.uint8)
    image = cv.imdecode(np_arr , cv.IMREAD_COLOR)

    return image


def encodeImageIntoBase64(image):
     _, img_encoded = cv.imencode('.jpg', image) 
     image = base64.b64encode(img_encoded).decode('utf-8')

     return image
    
