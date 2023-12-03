import sys,os
from CelebrityFaceDetection.logger import logging
from CelebrityFaceDetection.exception import AppException
from CelebrityFaceDetection.components.data_ingestion import DataIngestion
from CelebrityFaceDetection.components.model_trainer import ModelTrainer

from CelebrityFaceDetection.entity.config_entity import (DataIngestionConfig,ModelTrainerConfig)

from CelebrityFaceDetection.entity.artifacts_entity import (DataIngestionArtifact,ModelTrainerArtifact)
import numpy as np


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.model_trainer_config = ModelTrainerConfig()
        

    def start_data_ingestion(self)-> DataIngestionArtifact:
            try: 
                logging.info(
                    "Entered the start_data_ingestion method of TrainPipeline class"
                )
                logging.info("Getting the data from URL")

                data_ingestion = DataIngestion(
                    data_ingestion_config =  self.data_ingestion_config
                )

                data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
                logging.info("Got the data from URL")
                logging.info(
                    "Exited the start_data_ingestion method of TrainPipeline class"
                )
    
                return data_ingestion_artifact

            except Exception as e:
                raise AppException(e, sys)
            

    def start_model_trainer(self) -> ModelTrainerArtifact:
         
         try:
              
              model_trainer = ModelTrainer(model_trainer_config= self.model_trainer_config , dataset_directory = ModelTrainerConfig.dataset_path)

              model_trainer_artifact = model_trainer.initiate_model_trainer()       

              return model_trainer_artifact          



         except Exception as e:
              raise AppException(e, sys)
         

         
         

        

    def run_pipeline(self) -> None:
            
            try:
                
                #data_ingestion_artifact = self.start_data_ingestion() 

                model_trainer_artifact = self.start_model_trainer()         

            
            except Exception as e:
                raise AppException(e, sys)




