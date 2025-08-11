import os
import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
from src.logger import logging
from src.exception import CustomException  
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.utils.main_utils import MainUtils





@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join("artifacts")
    ingested_train_file_path: str = os.path.join(artifact_dir, "application_train.csv")
    transformed_train_file_path: str = os.path.join(artifact_dir, "train.npz")
    transformed_test_file_path: str = os.path.join(artifact_dir, "test.npz")
    transformed_train_csv_path: str = os.path.join(artifact_dir, "train_processed.csv")
    transformed_test_csv_path: str = os.path.join(artifact_dir, "test_processed.csv")
    transformed_object_file_path: str = os.path.join(artifact_dir, "preprocessor.pkl")





class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()

    def initiate_data_transformation(self):
        logging.info("Data Transformation started")
        try:
            df = pd.read_csv(self.config.ingested_train_file_path)
            logging.info(f"Data loaded from {self.config.ingested_train_file_path}")

            if 'SK_ID_CURR' in df .columns:
                df.drop(columns=['SK_ID_CURR'], inplace = True)
                logging.info("Dropped 'SK_ID_CURR' column from the dataset")

            X = df.drop(columns=['TARGET'])
            y = df['TARGET']
            logging.info("Features and target variable separated")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            logging.info("Data split into train and test sets")
            logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
            numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numeric_cols}")

            # Numerical Pipeline
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            #fit and transform training data
            X_train_num = numeric_pipeline.fit_transform(X_train[numeric_cols])
            X_test_num = numeric_pipeline.transform(X_test[numeric_cols])
            logging.info("Numerical features transformed using RobustScaler")

            # Categorical Pipeline
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
            logging.info("Categorical features transformed using one-hot encoding")

            # Align the columns of train and test sets
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
            logging.info("Aligned categorical features of train and test sets")

            # Combine transformed numerical and categorical features
            X_train_processed = np.hstack([X_train_num, X_train_cat.values])
            X_test_processed = np.hstack([X_test_num, X_test_cat.values])
            logging.info("Combined numerical and categorical features")

            """categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', self.utils.get_one_hot_encoder(categorical_cols))  # Placeholder for future encoding steps
            ])

            #fit and transform training data
            X_train[categorical_cols] = categorical_pipeline.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = categorical_pipeline.transform(X_test[categorical_cols])
            logging.info("Categorical features transformed using Encoder")"""

            
            # df.fillna(0, inplace=True)
            # logging.info("Missing values filled with 0")

            # df.to_csv(self.config.transform_train_csv_path, index=False)
            # logging.info(f"Transformed train data saved to {self.config.transform_train_csv_path}")

            # self.utils.save_object(self.config.transform_object_file_path, df)
            # logging.info(f"Transformation object saved to {self.config.transform_object_file_path}")

            # Save the transformed data
            np.savez(self.config.transformed_train_file_path, X=X_train_processed, y=y_train.values)
            np.savez(self.config.transformed_test_file_path, X=X_test_processed, y=y_test.values)
            logging.info(f"Transformed train data saved to {self.config.transformed_train_file_path}) and {self.config.transformed_test_file_path}")

            # Save the transformation object
            all_fetures_names = numeric_cols + list(X_train_cat.columns)
            train_df_out = pd.DataFrame(X_train_processed, columns=all_fetures_names)
            test_df_out = pd.DataFrame(X_test_processed, columns=all_fetures_names)
            train_df_out['TARGET'] = y_train.values
            test_df_out['TARGET'] = y_test.values
            train_df_out.to_csv(self.config.transformed_train_csv_path,index=False)
            test_df_out.to_csv(self.config.transformed_test_csv_path, index=False)

            # Save the transformation object
            preprocessor = {
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols,                
                'numeric_pipeline': numeric_pipeline,
                'categorical_columns': X_train_cat.columns.tolist()
            }
            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)
            logging.info(f"Transformation object saved at {self.config.transformed_object_file_path}")


            return(
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path,
                # self.config.transformed_train_csv_path,
                # self.config.transformed_test_csv_path    
                           
            )
        

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)


