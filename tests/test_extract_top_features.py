import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.extract_top_features import extract_and_save_top_features

if __name__ == "__main__":
    top_features, features_means = extract_and_save_top_features()
    print("The top features are:", top_features)



    # python tests\test_extract_top_features.py