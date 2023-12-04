from CelebrityFaceDetection.logger import logging
from CelebrityFaceDetection.pipeline.training_pipeline import TrainPipeline
import sys,os
from CelebrityFaceDetection.pipeline.training_pipeline import TrainPipeline
from CelebrityFaceDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from CelebrityFaceDetection.constant.application import APP_HOST, APP_PORT
import cv2 as cv
import numpy as np
os.environ['TF__CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import base64



app = Flask(__name__)
CORS(app)

facenet = FaceNet()
faces_embeddings = np.load(os.path.join('artifacts/models/faces_embeddings.npz'))
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open(os.path.join('artifacts/models/svm_model_160x160.pkl') , 'rb'))
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")



class ClientApp:
    def __init__(self):
        self.filename  = "inputImage.jpg"


@app.route('/train')
def trainRoute():

    obj = TrainPipeline()
    obj.run_pipeline()
    return 'Training Successfull!!'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict" , methods=["GET", "POST"])
@cross_origin()
def predictRoute():

    try:
        image_str = request.json['image']
        image = decodeImage(image_str)
       
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = haarcascade.detectMultiScale(gray_img , 1.3 , 5)

        """
        detectMultiScale(): This is a method provided by OpenCV to detect objects, in this case, faces, in an image. It takes several parameters:

        gray_img: This is the input image in grayscale. Haar Cascade classifiers typically work with grayscale images.
        1.3: This parameter is the scale factor. It determines how much the image size is reduced at each image scale. A smaller value will increase the detection time but may improve detection accuracy. A larger value speeds up detection but may reduce accuracy.
        5: This parameter is the minimum number of neighbors a region needs to have to be considered as a face. It helps filter out false positives. Increasing this value may also increase the quality of face detection but could miss some faces

        """

        for x,y,w,h in faces:
            img = rgb_img[y:y+h , x:x+w]
            img = cv.resize(img , (160,160))
            img = np.expand_dims(img , axis = 0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            encoder.transform(face_name)
            # final_name = encoder.inverse_transform(face_name)[0]
            cv.rectangle(image , (x,y) , (x+w , y+h) , (255,0,255) , 10)
            cv.putText(image , str(face_name) , (x,y-10) , cv.FONT_HERSHEY_SIMPLEX , 1 ,(0,0,255) , 3, cv.LINE_AA)  

            detected_image = encodeImageIntoBase64(image)

            result = {"image": detected_image}
            

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


cap = cv.VideoCapture(0)

@app.route("/live" , methods = ['GET'])
@cross_origin()
def predictLive():
    try:
        
        while cap.isOpened() :

            _ , frame = cap.read()
            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = haarcascade.detectMultiScale(gray_img , 1.3 , 5)
            
            for x,y,w,h in faces:
                img = rgb_img[y:y+h , x:x+w]
                img = cv.resize(img , (160,160))
                img = np.expand_dims(img , axis = 0)
                ypred = facenet.embeddings(img)
                face_name = model.predict(ypred)
                encoder.transform(face_name)
                # final_name = encoder.inverse_transform(face_name)[0]
                cv.rectangle(frame , (x,y) , (x+w , y+h) , (255,0,255) , 10)
                cv.putText(frame , str(face_name) , (x,y-10) , cv.FONT_HERSHEY_SIMPLEX , 1 ,(0,0,255) , 3, cv.LINE_AA)

                cv.imshow("Face Recognition:" , frame)


                if cv.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
                    cap.release()  # Release the camera
                    cv.destroyAllWindows()  # Close OpenCV windows
                    break


       
        
    except ValueError as val:
        print(val)
        return Response("value not found inside json data")

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)