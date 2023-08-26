import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.Data_Transformation_Config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:

            numerical_features = ['writing_score', 'reading_score']
            categorical_features= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)) ## Optional incase of categorical features
                ]
            )

            logging.info("Numerica columns scaling completed")

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', numerical_pipeline, numerical_features),
                    ('cat_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Started Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")

            logging.info("Preprocessing of data begins")
            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math_score'
            numerical_features = ['writing_score', 'reading_score']
            input_feature_train = train_df.drop(columns=[target_column], axis = 1)            
            target_feature_train = train_df[target_column]
            input_feature_test = test_df.drop(columns=[target_column], axis = 1)            
            target_feature_test = test_df[target_column]
            
            logging.info("Applying preprocessing object in dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)
            train_arr = np.c_[input_feature_train_arr,  np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr,  np.array(target_feature_test)]
            
            logging.info("Saved preprocessed object")
            save_object(
                file_path=self.Data_Transformation_Config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr, 
                test_arr, 
                self.Data_Transformation_Config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)