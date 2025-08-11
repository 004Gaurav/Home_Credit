# Importing required libraries
import os  # For file and folder operations
import sys  # To get system-specific parameters and functions
import pandas as pd  # For working with tabular data (CSV files)

# Importing constants and custom modules from your project structure
from src.constant import APPLICATION_TRAIN_PATH, APPLICATION_TEST_PATH  # File paths
from src.logger import logging  # Custom logging module to record events
from dataclasses import dataclass  # To create simple data classes
from src.exception import CustomException  # Custom error handling class

# Configuration class using @dataclass for automatic init method
@dataclass
class DataIngestionConfig:
    artifacts_folder: str = "artifacts"  # Folder to store output files
    train_file_path: str = "application_train.csv"  # Name of the output CSV file

# Main class to handle data ingestion
class DataIngestion:
    def __init__(self):
        # Creating a configuration object for data ingestion
        self.config = DataIngestionConfig()

    # Function to initiate the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")  # Log the start of the process
        try:
            # Create the artifacts folder if it doesn't exist
            os.makedirs(self.config.artifacts_folder, exist_ok=True)
            logging.info(f"created artifacts folder at {self.config.artifacts_folder}")
            
            # Define the destination path for saving the CSV file
            dst_path = os.path.join(self.config.artifacts_folder, self.config.train_file_path)
            logging.info(f"Copying data from {APPLICATION_TRAIN_PATH} to {dst_path}")
            
            # Read the dataset from the source path
            df = pd.read_csv(APPLICATION_TRAIN_PATH)
            
            # Save the DataFrame to the destination path as a CSV file
            df.to_csv(dst_path, index=False)
            logging.info(f"Data saved to {dst_path}")
            logging.info("Data Ingestion completed ")  # Log completion
        except Exception as e:
            # Log the error and raise a custom exception for handling
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)
