import os
import logging
import logging.config
import pandas as pd

from src.data.helpers.hash_list_of_dicts import hash_list_of_strings

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")

def _get_image_path_from_sample(sample):
    """
    Extracts the image path from a sample dictionary.
    Assumes the sample has a key "x-ray" that contains the image path.
    """
    if "x-ray" in sample:
        return sample["x-ray"]
    else:
        logger.error(f"Sample does not contain 'x-ray' key: {sample}")
        raise KeyError("Sample does not contain 'x-ray' key.")

def save_test_set_info(dataset_folder, test_samples, train_val_samples):
    """
    Save test set split information to a CSV file.
    
    Creates a DataFrame containing image paths and their corresponding test set labels,
    then saves it as a CSV file in the specified dataset folder. This helps ensure
    consistent test set splits across different runs.
    
    Args:
        dataset_folder (str): Path to the dataset folder where the CSV will be saved.
        test_samples (list): List of test samples to extract image paths from.
        train_val_samples (list): List of training/validation samples to extract image paths from.
    
    Returns:
        None
    
    Side Effects:
        - Creates a CSV file named 'test_set_split.csv' in the dataset_folder
        - Logs information about the saved file location
    """
    test_image_paths = [_get_image_path_from_sample(sample) for sample in test_samples]
    train_val_image_paths = [_get_image_path_from_sample(sample) for sample in train_val_samples]

    test_set_df = pd.DataFrame({
        "image_path": [path for path in test_image_paths],
        "test set": True
    })
    train_val_set_df = pd.DataFrame({
            "image_path": [path for path in train_val_image_paths],
            "test set": False
        })
    test_set_df = pd.concat([test_set_df, train_val_set_df], ignore_index=True)
    test_set_df.to_csv(os.path.join(dataset_folder, "test_set_split.csv"), index=False)
    logger.info(f"EnsureSameTestSet: Saved test set split to {os.path.join(dataset_folder, 'test_set_split.csv')}")


def save_test_set_hash(test_samples, dataset, hash_folder="datacache/"):
    """
    Save a hash of the test set image paths to a file for consistency validation.
    
    This function extracts image paths from test samples, computes a hash of these paths,
    and saves it to a text file. This allows for later verification that the same test
    set is being used across different runs or experiments.
    
    Args:
        test_samples: List of test samples containing image information
        dataset (str): Name of the dataset, used in the output filename
        hash_folder (str, optional): Directory to save the hash file. 
                                   Defaults to "datacache/"
    
    Returns:
        None
    
    Side Effects:
        - Creates a hash file named "{dataset}_test_set_hash.txt" in the specified folder
        - Logs information about the saved hash file location
    """
    test_image_paths = [_get_image_path_from_sample(sample) for sample in test_samples]
    test_set_hash = hash_list_of_strings(test_image_paths)
    with open(os.path.join(hash_folder, f"{dataset}_test_set_hash.txt"), "w") as f:
        f.write(test_set_hash)
    logger.info(f"EnsureSameTestSet: Saved test set hash to {os.path.join(hash_folder, f"{dataset}_test_set_hash.txt")}")

def load_test_and_train_split_from_saved_test_set_info(dataset_folder, dataset, data, hash_folder="datacache/"):
    """
    Load test and train data splits from a previously saved test set configuration.
    
    This function attempts to load a test set split from a CSV file and validates
    the integrity of the test set using stored hash values to ensure consistency
    across different runs.
    
    Args:
        dataset_folder (str): Path to the folder containing the dataset and split information.
        dataset (str): Name of the dataset, used for hash file naming.
        data (list): List of data samples, each containing an "x-ray" key with image path.
        hash_folder (str, optional): Path to folder for storing hash files. Defaults to "datacache/".
    
    Returns:
        tuple: A tuple containing (train_val_samples, test_samples) where:
            - train_val_samples (list): List of samples designated for training/validation
            - test_samples (list): List of samples designated for testing
            Returns (None, None) if the test set split file doesn't exist.
    
    Raises:
        ValueError: If the test set hash doesn't match the stored hash, indicating
                   the test set has been modified since creation.
    
    Notes:
        - Looks for 'test_set_split.csv' in the dataset_folder
        - Validates test set integrity using hash comparison
        - Creates hash file if one doesn't exist for future validation
    """
    if os.path.exists(os.path.join(dataset_folder, "test_set_split.csv")):
        logger.info(f"EnsureSameTestSet: Trying to load {os.path.join(dataset_folder, 'test_set_split.csv')}...")
        test_set_split_df = pd.read_csv(os.path.join(dataset_folder, "test_set_split.csv"))
        test_samples = [d for d in data if d["x-ray"] in test_set_split_df[test_set_split_df['test set'] == True]['image_path'].tolist()]
        train_val_samples = [d for d in data if d["x-ray"] in test_set_split_df[test_set_split_df['test set'] == False]['image_path'].tolist()]
        # if there is a hash stored for the test set, check if it matches the current test set
        cache_path = os.path.join(hash_folder, f"{dataset}_test_set_hash.txt")
        if os.path.exists(cache_path):
            if not check_test_set_hash(test_samples, cache_path):
                logger.critical(f"EnsureSameTestSet: The hash of the test set does not match the stored hash. This means that the test set has changed since the last time it was created.")
                raise ValueError(f"EnsureSameTestSet: The hash of the test set does not match the stored hash. This means that the test set has changed since the last time it was created.")
            else:
                logger.info(f"EnsureSameTestSet: The hash of the test set matches the stored hash. The test set is valid.")
        else:
            logger.warning(f"EnsureSameTestSet: No hash found for the test set. Nothing to compare against. Please make sure to store the hash of the test set after creating it.")
            save_test_set_hash(test_samples, dataset, hash_folder)

        logger.info(f"EnsureSameTestSet: Loaded existing test set split from {os.path.join(dataset_folder, 'test_set_split.csv')}")
        return train_val_samples, test_samples
    else:
        logger.critical(f"EnsureSameTestSet: {os.path.join(dataset_folder, 'test_set_split.csv')} does not exist.")
        return None, None
    
def check_test_set_hash(test_samples, cache_path):
    """Check if current test samples match the cached test set hash.

    Args:
        test_samples: List of test samples to verify
        cache_path: Path to file containing cached hash

    Returns:
        bool: True if hashes match, False otherwise
    """
    with open(cache_path, "r") as f:
        test_dicts_hash = f.read()
    current_test_dicts_hash = hash_list_of_strings([_get_image_path_from_sample(sample) for sample in test_samples])
    return test_dicts_hash == current_test_dicts_hash