
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


def start_training():

    try:
        # Step 1: Data Ingestion
        logging.info("Starting Data Ingestion")

        data_ingestion = DataIngestion()

        train_data, test_data = (
            data_ingestion.initiate_data_ingestion()
        )

        # Step 2: Data Transformation
        logging.info("Starting Data Transformation")

        data_transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        # Step 3: Model Training
        logging.info("Starting Model Training")

        model_trainer = ModelTrainer()

        r2_score = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )

        print("Training completed successfully!")
        print("Model R2 Score:", r2_score)
        print("Preprocessor:", preprocessor_path)
        print("Model saved at: artifacts/model.pkl")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    start_training()
