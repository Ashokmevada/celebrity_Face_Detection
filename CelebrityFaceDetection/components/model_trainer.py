import os,sys
import yaml
from CelebrityFaceDetection.utils.main_utils import read_yaml_file
from CelebrityFaceDetection.logger import logging
from CelebrityFaceDetection.exception import AppException
from CelebrityFaceDetection.entity.config_entity import ModelTrainerConfig
from CelebrityFaceDetection.entity.artifacts_entity import ModelTrainerArtifact
import mtcnn
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC






class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        dataset_directory
    ):
        self.model_trainer_config = model_trainer_config
        self.directory = dataset_directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = mtcnn.MTCNN()
    


    def extract_face(self , filename):

        img = cv.imread(filename)
        img = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img)
        x,y,w,h = results[0]['box']
        x , y = abs(x) , abs(y)
        face = img[y:y+h , x:x+w]
        face_arr = cv.resize(face , self.target_size)
        return face_arr
    

    def load_faces(self , dir):

        FACES = []
        for im_name in os.listdir(dir):

            print(im_name)

            try:
                
                path = dir + im_name
                print(path)
                single_face = self.extract_face(path)
                FACES.append(single_face)

            except Exception as e:
                pass
        return FACES
    

    
    def get_embedding(face_img):

        embedder = FaceNet()
        face_img = face_img.astype('float32') # 3D(160x160x3) # coinverting to float 32 for better performance
        face_img = np.expand_dims(face_img, axis=0) # expanding to 4d as as it required input shape for embeddings gives 1x160x160x3  1 is for batch size
        # 4D (Nonex160x160x3)
        yhat = embedder.embeddings(face_img) # this will return 1x1x512/128 as per the model
        return yhat[0] # 512D image (1x1x512)  we require only 1st feature but generally return yhat
    

    def load_classes(self):

        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded Successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        
        return np.asarray(self.X) , np.asarray(self.Y)
    
    def train_facenet(self):

        X , Y = self.load_classes()

            

        EMBEDDED_X = []

        for img in X:

            EMBEDDED_X.append(self.get_embedding(img))

        EMBEDDED_X = np.array(EMBEDDED_X)

        # compressed NumPy archive file (.npz format)multiple array save in one file

        np.savez_compressed('faces_embeddings.npz' , EMBEDDED_X , Y)

        return EMBEDDED_X






    def initiate_model_trainer(self , ) -> ModelTrainerArtifact:

        logging.info("Entered initiate model trainer method of ModelTraining Class")


        try:

            dataset_directory = ModelTrainerConfig.dataset_path                        
            
            EMBEDDED_X = self.train_facenet()

            face_embeddings = np.load("faces_embeddings.npz")

            Y = face_embeddings['arr_1']

            encoder = LabelEncoder()

            encoder.fit(Y)

            X_train, X_test, y_train, y_test = train_test_split(EMBEDDED_X , Y , test_size=0.2 , shuffle=True , random_state=85)

            SVM_model = SVC(kernel = 'linear' , probability=True)

            SVM_model.fit(X_train,y_train)

            import pickle

            with open('svm_model_160x160.pkl' , 'wb') as f:
                pickle.dump(SVM_model , f)


        except Exception as e:
            raise AppException(e , sys)