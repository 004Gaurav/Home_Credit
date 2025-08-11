import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.utils.main_utils import MainUtils




@dataclass
class ModelTrainerConfig:
        artifact_folder: str = os.path.join("artifacts")
        trained_model_path: str = os.path.join(artifact_folder, "model.pkl")
        expected_accuracy = 0.45
        model_config_file_path: str = os.path.join("config", "model.yaml")



class ModelTrainer:
        def __init__(self):
            self.config = ModelTrainerConfig()
            self.utils = MainUtils()
            self.models ={
                   "XGBClassifier": XGBClassifier(n_jobs=-1,verbosity=1),
                   "RandomForestClassifier": RandomForestClassifier(n_jobs=-1),
                    "GradientBoostingClassifier": GradientBoostingClassifier()
            }
            self.model_params_grid = self.utils.read_yaml_file(self.config.model_config_file_path)["model_selection"]["model"]

        def evaluate_models(self, X_train, y_train, X_test, y_test):
                logging.info("Evaluate models ..........")
                report ={}
                for name, model in self.models.items():
                      print(f"Training base model: {name}....")
                      logging.info(f"Training  {name} ....")
                      model.fit(X_train, y_train)
                      y_pred = model.predict(X_test)
                      score = accuracy_score(y_test, y_pred)
                      report[name] = score
                      logging.info(f"name accuracy: {score:.4f}")
                      print(f"{name} accuracy: {score:.4f}")
                logging.info(f"Model evaluvation report: {report}")
                print(f"Model evaluvation report: {report}")
                return report
        
        def finetune_best_model(self, model_name, model, X_train, y_train):
               print(f"Starting GridSearchCV for {model_name}...")
               logging.info(f"starting GridSearchCV for {model_name}...")
               param_grid = self.model_params_grid[model_name]["search_param_grid"]
               grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
               grid_search.fit(X_train, y_train)
               best_params = grid_search.best_params_
               print(f"Best params for {model_name}: {best_params}")
               logging.info(f"Best params for {model_name}: {best_params}")     
               model.set_params(**best_params)
               return model
        
        def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
                logging.info("Initiating model trainer...")
                try:
                        report = self.evaluate_models(X_train, y_train, X_test, y_test)
                        best_model_name = max(report, key=report.get)
                        best_model = self.models[best_model_name]
                        logging.info(f"Best model: {best_model_name} with accuracy: {report[best_model_name]:.4f}")

                        if report[best_model_name] < self.config.expected_accuracy:
                               raise CustomException(f"Model accuracy {report[best_model_name]:.4f} is below expected threshold {self.config.expected_accuracy:.4f}")
                        
                        best_model = self.finetune_best_model(best_model_name, best_model, X_train, y_train)


                        #save the trained model
                        self.utils.save_object(self.config.trained_model_path, best_model)
                        logging.info(f"Model saved at {self.config.trained_model_path}")
                        return self.config.trained_model_path
                
                except Exception as e:
                       logging.error(f"Error in model training: {str(e)}")
                       raise e
                
                        