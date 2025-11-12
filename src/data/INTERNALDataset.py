import logging
import logging.config
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.helpers.ensure_same_test_set import load_test_and_train_split_from_saved_test_set_info, save_test_set_hash, save_test_set_info
from src.data.helpers.encoding import encode_age, encode_anatomy_site, encode_sex
from src.data.helpers.internal_btxrd_combination import get_combined_anatomy_site_category


class INTERNALDataset:
    """
    INTERNALDataset
    A dataset class for handling internal x-ray data, including reading and parsing the raw data, splitting into 
    training/validation and test sets, and providing cross-validation splits, while ensuring no patient data leakage between train, validation and test sets.
    
    
    Args
    ----------
        path (str): Path to the dataset directory containing the images and metadata file.
        using_crops (bool): Indicates whether cropped images are used. Currently not supported.
        num_channels (int): Number of image channels (1 for grayscale, 3 for RGB).
    
    Attributes
    ----------
        train_val_dicts (list): List of dictionaries for training and validation samples.
        test_dicts (list): List of dictionaries for test samples.

    Methods
    ----------
        get_cv_splits() -> generator:
            Yields training and validation splits for cross-validation using stratified group k-fold.
        get_test_dicts() -> list:
            Returns the test set dictionaries.

    Raises
    ----------
        ValueError: If the `num_channels` parameter is not 1 or 3.
        AssertionError: If a patient has images in both training/validation and test sets.
    """

    def __init__(
        self,
        using_crops: bool,
        path: str,
        num_channels=3
    ):
        self.path = path
        self.using_crops = using_crops

        if not (num_channels == 1 or num_channels == 3):
            logger.error(
                f"INTERNALDataset: num_channels must be 1 or 3, but got {num_channels}"
            )
            raise ValueError(
                f"INTERNALDataset: num_channels must be 1 or 3, but got {num_channels}"
            )

        data_dicts = self._get_data_as_dict()
        self.train_val_dicts, self.test_dicts = self._split_test(data_dicts)
        

        logger.debug(f"INTERNALDatset: Number of training/vlaidation samples: {len(self.train_val_dicts)}")
        logger.debug(f"INTERNALDatset: Number of test samples: {len(self.test_dicts)}")
        logger.debug(
            f"INTERNALDatset: Ratio train_val/test: {len(self.train_val_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}/{len(self.test_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}"
        )

        # assert that no patient has images in two different sets
        train_patients = set([d["patient_number"] for d in self.train_val_dicts])
        test_patients = set([d["patient_number"] for d in self.test_dicts])
        assert (
            len(train_patients.intersection(test_patients)) == 0
        ), "There is at least one patient who has images in both train/val and test set"
        # patient_number was only needed for a proper split and needs to be removed now, s.t. it has the same structure as the BTXRD dataset
        for d in self.test_dicts:
            d.pop("patient_number")

        logger.info("INTERNALDataset: Successfully initialized the dataset")

    def _get_tumor_entity_from_sample(self, sample):
        return sample["entity"]
    
    def _get_anatomy_site_from_sample(self, sample):
        combined_category_anatomy_site = get_combined_anatomy_site_category([sample["localisation_1"]])
        return combined_category_anatomy_site

    def _get_data_as_dict(self):
        """
        Load and process patient data from Excel and CSV files into a list of dictionaries.
        
        This method reads two data sources:
        1. Tumor patients from 'included_patients.xlsx' 
        2. Non-tumor patients from 'healthy_subset_new_cleaned.csv'
        
        For each sample, it creates a standardized dictionary containing medical imaging
        data and metadata including demographics, anatomy information, and tumor status.
        
        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains:
                - dataset: Always "INTERNAL"
                - x-ray (str): Full path to the X-ray image file
                - image_path (str): Same as x-ray, kept for visualization purposes
                - tumor (int): Binary indicator (0 for healthy, 1 for tumor)
                - entity (str): Tumor entity type (from _get_tumor_entity_from_sample)
                - anatomy_site (str): Anatomical site in standardized format
                - anatomy_site_encoded: Encoded version of anatomy site
                - sex (str): Patient gender
                - sex_encoded (int): Encoded version of sex
                - age (int): Patient age
                - age_encoded (int): Encoded version of age
        
        Notes:
            - Image paths vary based on self.using_crops flag (cropped vs initial images)
            - German sex notation "W" (Weiblich) is converted to English "F" (Female)
            - All tumor patients have tumor=1, all healthy patients have tumor=0

        Raises:
            FileNotFoundError: If the required Excel or CSV files are not found.
            KeyError: If expected columns are missing in the data files.
        """
        included_patients_df = pd.read_excel(
            os.path.join(self.path, "included_patients.xlsx")
        )

        dicts = []
        for _, row in included_patients_df.iterrows():
            if self.using_crops:
                image_path = os.path.join(
                    self.path, "images_bounding_box_15_500_BILINEAR", row["image"]
                )
            else:
                image_path = os.path.join(self.path, "initial_images", row["image"])

            sex = "F" if row['sex'] == "W" else row["sex"] # turn german "W" for "Weiblich" into english "F" for "Female", the german "M" for "Männlich" is already "M" as in english for "Male"
            anatomy_site = self._get_anatomy_site_from_sample(row)

            dicts.append(
                {
                    "dataset": "INTERNAL",
                    "x-ray": image_path,
                    "image_path": image_path, # Just added for visualization purposes, to be able to see difference before and after transformations
                    "tumor": 1,  # all patients in the original internal dataset have tumors
                    # patien_number is an exception to having the same structure as the BTXRD dataset.
                    # The patient number is only used for splitting and is removed afterwards.
                    "patient_number": row["pat_nr"],
                    # metadata
                    "entity": self._get_tumor_entity_from_sample(row),
                    "anatomy_site": anatomy_site,
                    "anatomy_site_encoded": encode_anatomy_site(anatomy_site),
                    "sex": sex,
                    "sex_encoded": encode_sex(sex),
                    "age": row["age_initialdiagnosis"],
                    "age_encoded": encode_age(row["age_initialdiagnosis"]),
                }
            )

        included_non_tumor_patients_df = pd.read_csv(
            os.path.join(self.path, "healthy_subset_new_cleaned.csv")
        )

        for _, row in included_non_tumor_patients_df.iterrows():
            anatomy_site = row["anatomy_site"]
            sex = row["sex"]
            age = row["age"]
            dicts.append(
                {
                    "dataset": "INTERNAL",
                    "x-ray": row["file"],
                    "image_path": row["file"],  # Just added for visualization purposes, to be able to see difference before and after transformations
                    "tumor": 0,
                    "patient_number": row["patient_id"],
                    # metadata
                    "entity": "undefined",
                    "anatomy_site": anatomy_site,
                    "anatomy_site_encoded": encode_anatomy_site(anatomy_site),
                    "sex": sex,
                    "sex_encoded": encode_sex(sex),
                    "age": age,
                    "age_encoded": encode_age(age),
                }
            )

        return dicts

    def _split_test(self, data: list):
        """
        Split data into train/validation and test sets with stratification.
        
        Attempts to load existing test split first. If not found, creates a new stratified
        split based on tumor status and anatomy site, then saves the split for future use.
        
        Args:
            data (list): List of data samples to split
            
        Returns:
            tuple: (train_val_samples, test_samples) - Lists of samples for training/validation and testing
            
        Raises:
            FileNotFoundError: If existing test split file is not found
        """
        train_val_samples, test_samples = load_test_and_train_split_from_saved_test_set_info(dataset_folder=self.path, dataset='INTERNAL', data=data)
        if train_val_samples is None or test_samples is None:
            # to be save throw an error, if the test split is not already created (s.t. there is no way I can accidentally use a wrong test split)
            raise FileNotFoundError(f"INTERNAL: The file with the test set info does not exist. You can comment out this error throw to create a new one, but MAKE SURE THIS IS REALLY WHAT YOU WANT!")
            pass
        else:
            logger.info(f"INTERNALDataset: Using existing test set split.")
            return train_val_samples, test_samples
        logger.warning(f"INTERNALDataset: Creating test set split. ARE YOU SURE THIS IS WHAT YOU WANT? This will overwrite the existing test set split if it exists.")


        stratification_labels = [f"{d["tumor"]}, {d["anatomy_site"]}" for d in data]
        groups = [d['patient_number'] for d in data]

        # ---- Split: Train/Val (80%) vs Test (20%) ---- #
        sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)

        train_val_indices, test_indices = next(
            sgkf1.split(data, stratification_labels, groups)
        )
        train_val_dicts = [data[i] for i in train_val_indices]
        test_dicts = [data[i] for i in test_indices]

        # save to a csv file the test set split
        save_test_set_info(dataset_folder=self.path, test_samples=test_dicts, train_val_samples=train_val_dicts)
        logger.info(f"INTERNALDataset: Created new test set split and saved it.")

        save_test_set_hash(test_dicts, dataset='INTERNAL')
        logger.info(f"INTERNALDataset: Saved the hash of the test.")

        return train_val_dicts, test_dicts

    def get_cv_splits(self):
        """
        Generate cross-validation splits using grouped stratified K-fold.
        
        Creates 4-fold cross-validation splits with stratification based on tumor status
        and anatomy site.
        That the same patient is not in both train and validation dataset is ensured by the group.
        
        Yields:
            tuple: A tuple containing (train_dicts, val_dicts) where:
                - train_dicts (list): List of training data dictionaries for this fold
                - val_dicts (list): List of validation data dictionaries for this fold
        
        Note:
            Uses StratifiedGroupKFold with n_splits=4, shuffle=True, and random_state=42
            for reproducible splits.
        """
        stratification_labels = [f"{d['tumor']}, {d['anatomy_site']}" for d in self.train_val_dicts]
        groups = [d['patient_number'] for d in self.train_val_dicts]

        copy_train_val_dicts = [d.copy() for d in self.train_val_dicts] # copy the train_val_dicts, s.t. the function get_cv_splits can be called multiple times without changing the original train_val_dicts
        # remove the key value pair "patient_number" from the dicts, since it is only needed for splitting and there needs to be the same keys as in the BTXRD dataset
        for d in copy_train_val_dicts:
            d.pop("patient_number")

        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

        for train_indices, val_indices in sgkf.split(copy_train_val_dicts, stratification_labels, groups):
            train_dicts = [copy_train_val_dicts[i] for i in train_indices]
            val_dicts = [copy_train_val_dicts[i] for i in val_indices]

            yield train_dicts, val_dicts

    def get_test_dicts(self):
        return self.test_dicts


if __name__ == "__main__":
    path = os.environ.get("INTERNAL_DATASET_PATH")
    internal_dataset = INTERNALDataset(using_crops=False, path=path)
    for i, (train_dicts, val_dicts) in enumerate(
        internal_dataset.get_cv_splits()
    ):
        logger.info(f"Fold {i + 1}:")
        logger.info(f"Train dicts: {len(train_dicts)}")
        logger.info(f"Val dicts: {len(val_dicts)}")
    logger.info(f"First sample of train dicts: {train_dicts[0]}")

    # count the number of negative and positive samples in the training set
    train_df = pd.DataFrame(internal_dataset.train_val_dicts)
    train_tumor_counts = train_df['tumor'].value_counts().to_dict()
    train_entity_counts = train_df['entity'].value_counts().to_dict()
    train_anatomy_site_counts = train_df['anatomy_site'].value_counts().to_dict()

    logger.info(f"Train dataset tumor counts: {train_tumor_counts}")
    logger.info(f"Train dataset entity counts: {train_entity_counts}")
    logger.info(f"Train dataset anatomy site counts: {train_anatomy_site_counts}")

    # same for test set
    test_df = pd.DataFrame(internal_dataset.test_dicts)
    test_tumor_counts = test_df['tumor'].value_counts().to_dict()
    test_entity_counts = test_df['entity'].value_counts().to_dict()
    test_anatomy_site_counts = test_df['anatomy_site'].value_counts().to_dict()
    logger.info(f"Test dataset tumor counts: {test_tumor_counts}")
    logger.info(f"Test dataset entity counts: {test_entity_counts}")
    logger.info(f"Test dataset anatomy site counts: {test_anatomy_site_counts}")

    # start = time.perf_counter()
    # internaldataset = INTERNALDataset(using_crops=False)
    # end = time.perf_counter()

    # elapsed_ms = (end - start) * 1000
    # logger.info(f"INTERNAL Dataset: Time to load the dataset: {elapsed_ms:.2f} ms")

    # train_dataset = internaldataset.get_train_dataset()
    # val_dataset = internaldataset.get_val_dataset()
    # test_dataset = internaldataset.get_test_dataset()
    # logger.info(f"Training dataset length: {len(train_dataset)}")
    # logger.info(f"Fourth sample of training dataset: {train_dataset[3]}")
    # logger.info(
    #     f'Shape of X-Ray image of first sample: {train_dataset[0]["x-ray"].shape}'
    # )
    # logger.info(f'Entity of first sample: {train_dataset[0]["entity"]}')

    # train_df = pd.DataFrame(internaldataset.train_dicts)
    # val_df = pd.DataFrame(internaldataset.val_dicts)
    # test_df = pd.DataFrame(internaldataset.test_dicts)

    # # Entity counts
    # train_entity_counts = train_df['entity'].value_counts().to_dict()
    # val_entity_counts = val_df['entity'].value_counts().to_dict()
    # test_entity_counts = test_df['entity'].value_counts().to_dict()

    # logger.info(f"Train dataset entity counts: {train_entity_counts}")
    # logger.info(f"Validation dataset entity counts: {val_entity_counts}")
    # logger.info(f"Test dataset entity counts: {test_entity_counts}")

    # # Entity ratios
    # train_entity_ratios = train_df['entity'].value_counts(normalize=True).to_dict()
    # val_entity_ratios = val_df['entity'].value_counts(normalize=True).to_dict()
    # test_entity_ratios = test_df['entity'].value_counts(normalize=True).to_dict()

    # logger.info(f"Train dataset entity ratios: {train_entity_ratios}")
    # logger.info(f"Validation dataset entity ratios: {val_entity_ratios}")
    # logger.info(f"Test dataset entity ratios: {test_entity_ratios}")

    # # plot a bar chart with 3 bars for each entity showing the ratio of the entity in train, val and test dataset
    # import matplotlib.pyplot as plt
    # import numpy as np
    # all_entities = sorted(set(train_entity_ratios) | set(val_entity_ratios) | set(test_entity_ratios))

    # train_ratios = [train_entity_ratios.get(entity, 0) for entity in all_entities]
    # val_ratios = [val_entity_ratios.get(entity, 0) for entity in all_entities]
    # test_ratios = [test_entity_ratios.get(entity, 0) for entity in all_entities]

    # plt.figure(figsize=(12, 5))
    # bar_width = 0.25
    # index = np.arange(len(all_entities))

    # plt.bar(index, train_ratios, bar_width, label='Train')
    # plt.bar(index + bar_width, val_ratios, bar_width, label='Validation')
    # plt.bar(index + 2 * bar_width, test_ratios, bar_width, label='Test')

    # plt.xlabel('Tumor Entity')
    # plt.ylabel('Ratio')
    # plt.title('Ratio of Tumor Entities in Train, Validation and Test Datasets')
    # plt.xticks(index + bar_width, all_entities, rotation=45, ha='right')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("INTERNALDataset_entity_ratios.png")

    # width_greater_than_height = 0
    # height_greater_than_width = 0
    # equal = 0
    # for i, sample in enumerate(train_dataset):
    #     # the sample has dimension (n, width, height), count how often widht is greater than height and the other way around and also equal
    #     if sample["x-ray"].shape[1] > sample["x-ray"].shape[2]:
    #         width_greater_than_height += 1
    #     elif sample["x-ray"].shape[1] < sample["x-ray"].shape[2]:
    #         height_greater_than_width += 1
    #     else:
    #         equal += 1
    # logger.info(f'Number of samples with width greater than height: {width_greater_than_height}')
    # logger.info(f'Number of samples with height greater than width: {height_greater_than_width}')
    # logger.info(f'Number of samples with equal width and height: {equal}')

    # for i, sample in enumerate(dataset):
    #     if sample['x-ray'][3].min() == 255:
    #         print(f"Sample {i} has a min value of 255 in the 4th channel.")
    #     else:
    #         print(f"!!!################ Sample {i} does not have a min value of 255 in the 4th channel.")

    #     check whether the first three channels are equal
    #     if (sample['x-ray'][0] == sample['x-ray'][1]).all() and (sample['x-ray'][1] == sample['x-ray'][2]).all():
    #         print(f"Sample {i} has the first three channels equal.")
    #     else:
    #         print(f"!!!################ Sample {i} does not have the first three channels equal.")

    # plot the first 5 samples
    # import matplotlib.pyplot as plt
    # import numpy as np

    # fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # for i, sample in enumerate(train_dataset[:5]):
    #     image = sample["x-ray"][0].numpy()
    #     ax = axes[i]
    #     ax.imshow(image, cmap="gray")
    #     ax.axis("off")
    #     ax.set_title(f"Sample {i}")

    # plt.tight_layout()
    # plt.savefig("INTERNALDataset_samples.png")
