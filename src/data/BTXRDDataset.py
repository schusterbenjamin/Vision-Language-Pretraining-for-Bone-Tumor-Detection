import logging
import logging.config
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.helpers.ensure_same_test_set import load_test_and_train_split_from_saved_test_set_info, save_test_set_hash, save_test_set_info
from src.data.helpers.encoding import encode_age, encode_anatomy_site, encode_sex
from src.data.helpers.internal_btxrd_combination import get_combined_anatomy_site_category

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class BTXRDDataset:
    """
    BTXRDDataset
    A dataset class for handling btxrd x-ray data, including reading and parsing the raw data, splitting into 
    training/validation and test sets, and providing cross-validation splits.

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
    """

    def __init__(
        self,
        using_crops: bool,
        path: str,
        num_channels=3,
    ):
        self.path = path
        self.using_crops = using_crops
        if using_crops:
            raise NotImplementedError(
                "Crops are not implemented yet. Please use the full images."
            )

        if not (num_channels == 1 or num_channels == 3):
            raise ValueError(
                f"INTERNALDataset: num_channels must be 1 or 3, but got {num_channels}"
            )

        data_dicts = self._get_data_as_dict()
        test_size = int(0.2 * len(data_dicts))
        self.train_val_dicts, self.test_dicts = self._split_test(data_dicts, test_size)

        logger.debug(f"BTXRDDataset: Number of training/validation samples: {len(self.train_val_dicts)}")
        logger.debug(f"BTXRDDataset: Number of test samples: {len(self.test_dicts)}")
        logger.debug(
            f"BTXRDDataset: Ratio train+val/test: {len(self.train_val_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}/{len(self.test_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}"
        )

        logger.info("BTXRDDataset: Successfully initialized the dataset")


    def _get_tumor_entity_from_sample(self, sample):
        tumor_types = ['osteochondroma',
       'multiple osteochondromas', 'simple bone cyst', 'giant cell tumor',
       'osteofibroma', 'synovial osteochondroma', 'other bt', 'osteosarcoma',
       'other mt']
        # check for the columns tumor_types, which one is set to 1
        selected = sample[tumor_types] == 1
        # There is one sample where two tumor types are set to 1, so I select the first one
        return sample[tumor_types][selected].index[0] if selected.any() else 'undefined'

    def _get_anatomy_site_from_sample(self, sample):
        anatomy_sites = ['hand', 'ulna', 'radius',
       'humerus', 'foot', 'tibia', 'fibula', 'femur', 'hip bone',
       'ankle-joint', 'knee-joint', 'hip-joint', 'wrist-joint', 'elbow-joint',
       'shoulder-joint']
        # check for the columns anatomy_sites, which one is set to 1
        selected = sample[anatomy_sites] == 1
        combined_category_anatomy_site = get_combined_anatomy_site_category(
            sample[anatomy_sites].index[selected].tolist()
        )
        return combined_category_anatomy_site

    def _get_data_as_dict(self):
        """
        Load and process BTXRD dataset data into a list of dictionaries.
        
        This method reads the main dataset Excel file and the healthy anatomy sites mapping file,
        then processes each row to create a standardized dictionary format for the dataset.
        For healthy samples (tumor=0), anatomy sites are retrieved from a separate mapping file.
        For tumor samples, anatomy sites are processed through internal translation methods.
        
        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains:
                - dataset (str): Always "BTXRD"
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
        
        Raises:
            FileNotFoundError: If dataset.xlsx or BTXRD_healthy_anatomy_sites.xlsx files are not found
            KeyError: If required columns are missing from the Excel files
            IndexError: If anatomy site mapping is not found for a healthy sample
        """
        btxrd_df = pd.read_excel(os.path.join(self.path, "dataset.xlsx"))
        btxrd_healthy_anatomy_sites_df = pd.read_excel(os.path.join(self.path, "BTXRD_healthy_anatomy_sites.xlsx"))
        btxrd_healthy_anatomy_sites_df['choice'] = btxrd_healthy_anatomy_sites_df['choice'].str.lower() # make them lower case to match the anatomy sites in the joint format

        dicts = []
        for _, row in btxrd_df.iterrows():
            image_path = os.path.join(self.path, "images", row["image_id"])

            if row['tumor'] == 0: # if it is a healthy sample, we need to get the anatomy site from the healthy anatomy sites dataframe
                anatomy_site = btxrd_healthy_anatomy_sites_df[btxrd_healthy_anatomy_sites_df['image_file'] == row['image_id']]['choice'].values[0]
            else: # if it is not a tumor, we need to translate the anatomy site into a shared format with INTERNAL dataset, this is not necessary for the non-tumor samples, since we already labeled them in the shared format
                anatomy_site = self._get_anatomy_site_from_sample(row)

            dicts.append(
                {
                    "dataset": "BTXRD",
                    "x-ray": image_path,
                    "image_path": image_path, # Just added for visualization purposes, to be able to see difference before and after transformations
                    "tumor": row["tumor"],
                    # metadata
                    "entity": self._get_tumor_entity_from_sample(row),
                    "anatomy_site": anatomy_site,
                    "anatomy_site_encoded": encode_anatomy_site(anatomy_site),
                    "sex": row['gender'],
                    "sex_encoded": encode_sex(row['gender']),
                    "age": row['age'],
                    "age_encoded": encode_age(row['age']),
                }
            )

        return dicts

    def _split_test(self, data: list, test_size: int):
        """
        Split data into train/validation and test sets with stratification.
        
        Attempts to load existing test split first. If not found, creates a new stratified
        split based on tumor status and anatomy site, then saves the split for future use.
        
        Args:
            data (list): List of data samples to split
            test_size (int): Number of samples for test set
            
        Returns:
            tuple: (train_val_samples, test_samples) - Lists of samples for training/validation and testing
            
        Raises:
            FileNotFoundError: If existing test split file is not found
        """
        train_val_samples, test_samples = load_test_and_train_split_from_saved_test_set_info(dataset_folder=self.path, dataset='BTXRD', data=data)
        if train_val_samples is None or test_samples is None:
            # to be save throw an error, if the test split is not already created (s.t. there is no way I can accidentally use a wrong test split)
            raise FileNotFoundError(f"BTXRD: The file with the test set info does not exist. You can comment out this error throw to create a new one, but MAKE SURE THIS IS REALLY WHAT YOU WANT!")
            pass
        else:
            logger.info(f"BTXRDDataset: Using existing test set split.")
            return train_val_samples, test_samples
        logger.warning(f"BTXRDDataset: Creating test set split. ARE YOU SURE THIS IS WHAT YOU WANT? This will overwrite the existing test set split if it exists.")

        # Note: I can later add e.g. the anatomy site to the stratification labels
        # The idea is to combine all variables into a tuple. That way when stratifying over it, it will be as stratifying over multiple variables.
        # But that does not work for continuos variables.

        # use 'undefined' anatomy site for stratification labels to have the same test split as before anatomy site got annotated for healthy samples
        stratification_labels = [(d["tumor"], d["anatomy_site"]) if d["tumor"] == 1 else (d["tumor"], "undefined") for d in data]

        train_val_samples, test_samples, _, _ = train_test_split(
            data,
            stratification_labels,
            test_size=test_size,
            stratify=stratification_labels,
            random_state=42,
        )

        # save to a csv file the test set split
        save_test_set_info(dataset_folder=self.path, test_samples=test_samples, train_val_samples=train_val_samples)
        logger.info(f"BTXRDDataset: Created new test set split and saved it.")

        save_test_set_hash(test_samples, dataset='BTXRD')
        logger.info(f"BTXRDDataset: Saved the hash of the test.")

        return train_val_samples, test_samples

    def get_cv_splits(self):
        """
        Generate cross-validation splits using stratified K-fold.
        
        Creates 4-fold cross-validation splits with stratification based on tumor status
        and anatomy site. For healthy samples (tumor=0), uses 'undefined' as anatomy
        site to maintain consistency with previous validation splits before anatomy
        site annotation was added for healthy samples.
        
        Yields:
            tuple: A tuple containing (train_dicts, val_dicts) where:
                - train_dicts (list): List of training data dictionaries for this fold
                - val_dicts (list): List of validation data dictionaries for this fold
        
        Note:
            Uses StratifiedKFold with n_splits=4, shuffle=True, and random_state=42
            for reproducible splits.
        """
        # use 'undefined' anatomy site for stratification labels to have the same val split as before the anatomy site got annotated for healthy samples
        stratification_labels = [f"{d["tumor"]}, {d["anatomy_site"]})" if d["tumor"] == 1 else f"{d["tumor"]}, undefined" for d in self.train_val_dicts]

        sgkf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        for train_indices, val_indices in sgkf.split(self.train_val_dicts, stratification_labels):
            train_dicts = [self.train_val_dicts[i] for i in train_indices]
            val_dicts = [self.train_val_dicts[i] for i in val_indices]

            yield train_dicts, val_dicts

    def get_test_dicts(self):
        return self.test_dicts


if __name__ == "__main__":
    path = os.environ.get("BTXRD_DATASET_PATH")
    btxrd_dataset = BTXRDDataset(using_crops=False, path=path)
    for i, (train_dicts, val_dicts) in enumerate(
        btxrd_dataset.get_cv_splits()
    ):
        logger.info(f"Fold {i}:")
        logger.info(f"Train dicts: {len(train_dicts)}")
        logger.info(f"Val dicts: {len(val_dicts)}")

    # print first sample
    logger.info(f"First sample of train dicts: {train_dicts[0]}")

    # start = time.perf_counter()
    # dataset = BTXRDDataset(using_crops=False)
    # end = time.perf_counter()

    # elapsed_ms = (end - start) * 1000
    # logger.info(f"BTXRDDataset: Time to load the dataset: {elapsed_ms:.2f} ms")

    # train_dataset = dataset.get_train_dataset()
    # val_dataset = dataset.get_val_dataset()
    # test_dataset = dataset.get_test_dataset()

    # logger.info(f"Train dataset length: {len(train_dataset)} samples")
    # logger.info(f"First sample of training dataset: {train_dataset[0]}")
    # logger.info(
    #     f'Shape of X-Ray image of first sample: {train_dataset[0]["x-ray"].shape}'
    # )
    # logger.info(f'Tumor Entity of first sample: {train_dataset[0]["entity"]}')
    
    # # for each dataset, count how often each tumor entity occurs and what the ratio is
    # train_entities = [sample["entity"] for sample in train_dataset]
    # val_entities = [sample["entity"] for sample in val_dataset]
    # test_entities = [sample["entity"] for sample in test_dataset]
    # train_entity_counts = {entity: train_entities.count(entity) for entity in set(train_entities)}
    # val_entity_counts = {entity: val_entities.count(entity) for entity in set(val_entities)}
    # test_entity_counts = {entity: test_entities.count(entity) for entity in set(test_entities)}

    # logger.info(f"Train dataset entity counts: {train_entity_counts}")
    # logger.info(f"Validation dataset entity counts: {val_entity_counts}")
    # logger.info(f"Test dataset entity counts: {test_entity_counts}")

    # train_entity_ratios = {entity: count / len(train_dataset) for entity, count in train_entity_counts.items()}
    # val_entity_ratios = {entity: count / len(val_dataset) for entity, count in val_entity_counts.items()}
    # test_entity_ratios = {entity: count / len(test_dataset) for entity, count in test_entity_counts.items()}

    # # plot a bar chart with 3 bars for each entity showing the ratio of the entity in train, val and test dataset
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.figure(figsize=(10, 5))
    # bar_width = 0.25
    # index = np.arange(len(train_entity_ratios))
    # bar1 = plt.bar(index, train_entity_ratios.values(), bar_width, label='Train')
    # bar2 = plt.bar(index + bar_width, val_entity_ratios.values(), bar_width, label='Validation')
    # bar3 = plt.bar(index + 2 * bar_width, test_entity_ratios.values(), bar_width, label='Test')
    # plt.xlabel('Tumor Entity')
    # plt.ylabel('Ratio')
    # plt.title('Ratio of Tumor Entities in Train, Validation and Test Datasets')
    # plt.xticks(index + bar_width, train_entity_ratios.keys())
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("BTXRDDataset_entity_ratios.png")

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
    # logger.info(
    #     f"Number of samples with width greater than height: {width_greater_than_height}"
    # )
    # logger.info(
    #     f"Number of samples with height greater than width: {height_greater_than_width}"
    # )
    # logger.info(f"Number of samples with equal width and height: {equal}")


    # fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # for i, sample in enumerate(train_dataset[:5]):
    #     image = sample["x-ray"][0].numpy()
    #     ax = axes[i]
    #     ax.imshow(image, cmap="gray")
    #     ax.axis("off")
    #     ax.set_title(f"Sample {i}")

    # plt.tight_layout()
    # plt.savefig("BTXRDDataset_samples.png")
