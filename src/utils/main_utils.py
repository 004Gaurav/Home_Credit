import sys
import os
import pickle
import yaml
import pandas as pd
from typing import Dict, Any

from src.constant import *
from src.exception import CustomException
from src.logger import logging

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, file_path: str) -> dict:
        """
        Reads a YAML file and returns its content as a dictionary.
        """
        try:
            with open(file_path, 'rb') as yaml_file:
                content = yaml.safe_load(yaml_file)
            return content
        except Exception as e:
            logging.error(f"Error reading YAML file at {file_path}")
            raise CustomException(e, sys) from e
        
    def read_scehema_config_file(self) -> dict:
        """Reads the schema configuration file and returns its content as a dictionary."""
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))
            return schema_config
        except Exception as e:
            logging.error("Error reading schema configuration file.")
            raise CustomException(e, sys) from e
        
    @staticmethod
    def save_object(file_path: str, obj: Any) -> None:
        """
        Saves an object to a file using pickle.
        """
        logging.info(f"Saving object at {file_path}")
        
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)
            logging.info(f"Object saved at {file_path}")
        except Exception as e:
            logging.error(f"Error saving object at {file_path}")
            raise CustomException(e, sys) from e
        
    @staticmethod
    def load_object(file_path: str) -> Any:
        """
        Loads an object from a file using pickle.
        """
        logging.info(f"Loading object from {file_path}")
        
        try:
            with open(file_path, 'rb') as file:
                obj = pickle.load(file)
            logging.info(f"Object loaded from {file_path}")
            return obj
        except Exception as e:
            logging.error(f"Error loading object from {file_path}")
            raise CustomException(e, sys) from e