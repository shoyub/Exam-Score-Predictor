import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def main():
    try:
        logging.info("Starting training pipeline...")
        
        # Step 1: Data Ingestion
        logging.info("Step 1: Data Ingestion initiated")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train: {train_data_path}, Test: {test_data_path}")
        
        # Step 2: Data Transformation
        logging.info("Step 2: Data Transformation initiated")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation completed")
        
        # Step 3: Model Training
        logging.info("Step 3: Model Training initiated")
        model_trainer = ModelTrainer()
        best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed. Best Model: {best_model_name}, Score: {best_model_score}")
        
        print("\n" + "="*60)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"🤖 Best Model: {best_model_name}")
        print(f"📊 Model Score (R² Score): {best_model_score:.4f}")
        print("="*60 + "\n")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
