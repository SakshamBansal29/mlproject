from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_ingestion import DataIngestionConfig, DataIngestion


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    modeltrainer = ModelTrainer()
    r_square = modeltrainer.initiate_model_trainer(train_arr, test_arr)  

    print(r_square)
        