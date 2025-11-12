import logging
import logging.config
import os
import random
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.helpers.ensure_same_test_set import check_test_set_hash, load_test_and_train_split_from_saved_test_set_info, save_test_set_hash, save_test_set_info

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class MURADataset:
    """
    A dataset class for loading and processing MURA X-ray images with captions.
    This class handles loading X-ray images from the MURA dataset, assigns appropriate
    captions based on anatomy site and abnormality labels, and provides cross-validation
    splits.

    Args
    ----------
        path (str): Path to the MURA dataset directory
        captions_path (str): Path to the captions CSV file

    Attributes
    ----------
        train_val_dicts (list): Training and validation data dictionaries
        test_dicts (list): Test data dictionaries
        train_caption_ids (list): Corresponding caption IDs for each sample in the training set

    Methods
    ----------
        get_cv_splits(): Yield training and validation splits for cross-validation using stratified group k-fold.
        get_test_dicts(): Return the test set dictionaries.
    """
    def __init__(
        self,
        path: str,
        captions_path="/mnt/nfs/homedirs/benjamins/project/res/data/pretrain/captions.csv"
    ):
        self.path = os.path.expanduser(path)

        self.captions_df = pd.read_csv(captions_path)
        self.caption_mapping_state = {}

        self.train_val_dicts, self.train_val_caption_ids, self.test_dicts = self._get_data_as_dict()

        logger.debug(f"MURADataset: Number of training/validation samples: {len(self.train_val_dicts)}")
        logger.debug(f"MURADataset: Number of test samples: {len(self.test_dicts)}")
        logger.debug(
            f"MURADataset: Ratio train+val/test: {len(self.train_val_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}/{len(self.test_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}"
        )

        # assert that no patient has images in two different sets
        train_patients = set([d["patient_id"] for d in self.train_val_dicts])
        test_patients = set([d["patient_id"] for d in self.test_dicts])
        assert (
            len(train_patients.intersection(test_patients)) == 0
        ), "There is at least one patient who has images in both train/val and test set"
        # patient_id was only needed for a proper split and needs to be removed now, s.t. it has the same structure as the LERA dataset
        # for train_val dicts it gets removed after the split, since it is needed for the split it is not removed here
        for d in self.test_dicts:
            d.pop("patient_id")

        logger.info("MURADataset: Successfully initialized the dataset")

    def _get_caption(self, anatomy_site, label):
        possible_captions = self.captions_df[
            (self.captions_df["anatomy_site"] == anatomy_site)
            & (self.captions_df["abnormality_label"] == label)
        ]["caption"].tolist()

        # the last index for that anatomy site and label combination is stored. This way, we "cycle" through the captions and assign them
        # check if there is an entry f"{anatomy_site}-{label}" in the caption_mapping_state
        if f"{anatomy_site}-{label}" in self.caption_mapping_state:
            index = self.caption_mapping_state[f"{anatomy_site}-{label}"] + 1
            if index > len(possible_captions) - 1:
                index = 0
            self.caption_mapping_state[f"{anatomy_site}-{label}"] = index
            caption = possible_captions[index]
        else:
            index = 0
            self.caption_mapping_state[f"{anatomy_site}-{label}"] = index
            caption = possible_captions[index]

        # also get the index of the caption in the entire dataframe
        # Find the index of this caption in the entire captions dataframe
        caption_idx = self.captions_df[
            (self.captions_df["anatomy_site"] == anatomy_site) &
            (self.captions_df["abnormality_label"] == label) &
            (self.captions_df["caption"] == caption)
        ].index.tolist()

        if caption_idx:
            caption_idx = caption_idx[0]  # Take the first match if multiple exist

            if isinstance(caption_idx, list) and len(caption_idx) > 1:
                raise ValueError(
                    f"Multiple captions found for anatomy site '{anatomy_site}', label '{label}' and caption {caption}. Please ensure unique captions."
                )
        else:
            raise ValueError(
                f"Caption '{caption}' for anatomy site '{anatomy_site}' and label '{label}' not found in captions dataframe."
            )

        return caption, caption_idx

    def _get_data_as_dict(self):
        """
        Load and process MURA dataset files to create structured dictionaries for training and validation data.
        This method reads CSV files containing image paths and labels for both training and validation sets,
        extracts metadata (anatomy site, patient ID, study number), merges the data, and creates dictionaries
        suitable for dataset loading. It also assigns generated captions to each image and validates the test set
        integrity using hash verification.
        Returns:
            tuple: A tuple containing three elements:
                - train_dicts (list): List of dictionaries containing training data with keys:
                    - "dataset": Dataset name (str)
                    - "x-ray": Full image path (str)
                    - "image_path": Full image path (str) (for visualization purposes)
                    - "label": Abnormality label (str)
                    - "anatomy_site": Extracted anatomy site from image path (str)
                    - "caption": Generated caption based on anatomy site and label (str)
                    - "patient_id": Extracted patient ID from file path (str)
                - train_caption_ids (list): List of caption IDs corresponding to training data
                - val_dicts (list): List of dictionaries containing validation data with same structure
                    as train_dicts
        Raises:
            ValueError: If the test set hash validation fails, indicating the test set has been modified
        Notes:
            - Expects CSV files: train_labeled_studies.csv, train_image_paths.csv, 
            valid_labeled_studies.csv, valid_image_paths.csv in self.path
            - Extracts anatomy site from image paths using 'XR_' delimiter
            - Extracts patient ID and study number from file paths
            - Creates full image paths by combining self.path with relative paths
            - Performs hash-based validation of test set consistency
            - Saves test set hash if none exists for future validation
        """
       
        train_labels_file_path = os.path.join(self.path, "train_labeled_studies.csv")
        train_image_path_file_path = os.path.join(self.path, "train_image_paths.csv")
        validation_labels_file_path = os.path.join(self.path, "valid_labeled_studies.csv")
        validation_image_path_file_path = os.path.join(self.path, "valid_image_paths.csv")
        train_labels_file_path = os.path.expanduser(train_labels_file_path)
        train_image_path_file_path = os.path.expanduser(train_image_path_file_path)
        validation_labels_file_path = os.path.expanduser(validation_labels_file_path)
        validation_image_path_file_path = os.path.expanduser(validation_image_path_file_path)

        train_labels_df = pd.read_csv(train_labels_file_path, header=None)
        train_image_paths_df = pd.read_csv(train_image_path_file_path, header=None)
        validation_labels_df = pd.read_csv(validation_labels_file_path, header=None)
        validation_image_path_df = pd.read_csv(validation_image_path_file_path, header=None)

        # Extract Label
        def extract_anatomy_site(path):
            try:
                # Find string after "XR_" and before next "/"
                parts = path.split('XR_')[1].split('/')
                return parts[0]
            except:
                print(f"Error processing path: {path}")
                return 'Unknown'

        # Extract Labels for training and validation sets
        train_image_paths_df['anatomy_site'] = train_image_paths_df.iloc[:, 0].apply(extract_anatomy_site)
        validation_image_path_df['anatomy_site'] = validation_image_path_df.iloc[:, 0].apply(extract_anatomy_site)

        # Function to extract study information
        def extract_study_info(path):
            try:
                # Split by patient and study
                parts = path.split('patient')[1]
                # Patient ID is before the next /
                patient_id = parts.split('/')[0]
                # Study part is after the patient_id/
                study_part = parts.split('/')[1]
                # Extract study number
                study_number = study_part.split('_')[0].replace('study', '')
                return patient_id, study_number
            except:
                print(f"Error processing path: {path}")
                return 'Unknown', 'Unknown'
            
        # need to create a total path from the given path
        def create_total_image_path(path):
            path = os.path.join(*(path.split(os.sep)[1:])) # the path in the datafram has "MURA-v1.1/" but the base path has it as well, so before combing them, do I have to remove it
            return os.path.join(self.path, path)

        # Create DataFrames to store patient and study information
        train_image_paths_df[['patient_id', 'study_number']] = train_image_paths_df.iloc[:, 0].apply(lambda x: pd.Series(extract_study_info(x)))
        train_image_paths_df.columns = ['image_path', 'anatomy_site', 'patient_id', 'study_number']
        validation_image_path_df[['patient_id', 'study_number']] = validation_image_path_df.iloc[:, 0].apply(lambda x: pd.Series(extract_study_info(x)))
        validation_image_path_df.columns = ['image_path', 'anatomy_site', 'patient_id', 'study_number']
        train_image_paths_df['study_path'] = train_image_paths_df['image_path'].apply(lambda x: '/'.join(x.split('/')[:-1]) + '/')
        validation_image_path_df['study_path'] = validation_image_path_df['image_path'].apply(lambda x: '/'.join(x.split('/')[:-1]) + '/')
        train_image_paths_df['image_path'] = train_image_paths_df['image_path'].apply(create_total_image_path)
        validation_image_path_df['image_path'] = validation_image_path_df['image_path'].apply(create_total_image_path)
        logger.debug(f"MURADataset: example of a train image path: {train_image_paths_df['image_path'].iloc[0]}")

        # Merge with labels
        train_labels_df.columns = ['study_path', 'label']
        validation_labels_df.columns = ['study_path', 'label']

        train_image_paths_df = train_image_paths_df.merge(train_labels_df, on='study_path', how='left')
        validation_image_path_df = validation_image_path_df.merge(validation_labels_df, on='study_path', how='left')

        train_dicts = []
        train_caption_ids = []
        for _, row in train_image_paths_df.iterrows():
            caption, caption_id = self._get_caption(row["anatomy_site"], row["label"])
            train_dicts.append(
                {
                    "dataset": "MURA",
                    "x-ray": row["image_path"],
                    "image_path": row["image_path"], # Just added for visualization purposes, to be able to see difference before and after transformations
                    "label": row["label"],
                    # metadata
                    "anatomy_site": row['anatomy_site'],
                    "caption": caption,
                    # patient_id is an exception to having the same structure as the LERA dataset.
                    # The patient id is only used for splitting and is removed afterwards.
                    "patient_id": row['patient_id'],
                }
            )
            train_caption_ids.append(caption_id)

        val_dicts = []
        for _, row in validation_image_path_df.iterrows():
            caption, _ = self._get_caption(row["anatomy_site"], row["label"])
            val_dicts.append(
                {
                    "dataset": "MURA",
                    "x-ray": row["image_path"],
                    "image_path": row["image_path"], # Just added for visualization purposes, to be able to see difference before and after transformations
                    "label": row["label"],
                    # metadata
                    "anatomy_site": row['anatomy_site'],
                    "caption": caption,
                    # patient_id is an exception to having the same structure as the LERA dataset.
                    # The patient id is only used for splitting and is removed afterwards.
                    "patient_id": row['patient_id'],
                }
            )

        cache_path = os.path.join("datacache/", f"MURA_test_set_hash.txt")
        if os.path.exists(cache_path):
            if not check_test_set_hash(val_dicts, cache_path):
                logger.critical(f"MURADataset: The hash of the test set does not match the stored hash. This means that the test set has changed since the last time it was created.")
                raise ValueError(f"MURADataset: The hash of the test set does not match the stored hash. This means that the test set has changed since the last time it was created.")
            else:
                logger.info(f"MURADataset: The hash of the test set matches the stored hash. The test set is valid.")
        else:
            logger.warning(f"MURADataset: No hash found for the test set. Nothing to compare against. Please make sure to store the hash of the test set after creating it.")
            save_test_set_hash(val_dicts, "MURA", "datacache/")

        return train_dicts, train_caption_ids, val_dicts

    def get_cv_splits(self):
        """
        Yields 5 training and validation splits for cross-validation using stratified group k-fold.
        
        Yields
        ----------
            train_dicts (list): Training data dictionaries for the current fold
            train_caption_ids (list): Caption IDs for the training data in the current fold
            val_dicts (list): Validation data dictionaries for the current fold
            val_caption_ids (list): Caption IDs for the validation data in the current fold
        """
        stratification_labels = [f"{d['label']}, {d['anatomy_site']}" for d in self.train_val_dicts]
        groups = [d['patient_id'] for d in self.train_val_dicts]

        # remove the key value pair "patient_id" from the dicts, since it is only needed for splitting and there needs to be the same keys as in the LERA dataset
        # but do this on a copy, s.t. this function can be called multiple times without side effects
        train_val_dicts = [d.copy() for d in self.train_val_dicts]
        for d in train_val_dicts:
            d.pop("patient_id")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        for train_indices, val_indices in sgkf.split(train_val_dicts, stratification_labels, groups):
            train_dicts = [train_val_dicts[i] for i in train_indices]
            val_dicts = [train_val_dicts[i] for i in val_indices]
            train_caption_ids = [self.train_val_caption_ids[i] for i in train_indices]
            val_caption_ids = [self.train_val_caption_ids[i] for i in val_indices]

            yield train_dicts, train_caption_ids, val_dicts, val_caption_ids

    def get_test_dicts(self):
        """
        Returns the test set dictionaries.

        Returns
        ----------
            test_dicts (list): test data dictionaries"""
        return self.test_dicts


if __name__ == "__main__":
    path = os.environ.get("MURA_DATASET_PATH")
    mura_dataset = MURADataset(path=path)
    len_test_dicts = len(mura_dataset.get_test_dicts())
    for i, (train_dicts, _, val_dicts, _) in enumerate(
        mura_dataset.get_cv_splits()
    ):
        len_train_dicts = len(train_dicts)
        len_val_dicts = len(val_dicts)
        logger.info(f"Fold {i}:")
        logger.info(f"Train dicts: {len_train_dicts}, Validation dicts: {len_val_dicts}, Test dicts: {len_test_dicts}")
        logger.info(f"Ratio train/val/test: {len_train_dicts/(len_train_dicts+len_val_dicts+len_test_dicts):.2f}/{len_val_dicts/(len_train_dicts+len_val_dicts+len_test_dicts):.2f}/{len_test_dicts/(len_train_dicts+len_val_dicts+len_test_dicts):.2f}")

    logger.debug(f"MURADataset: First training sample: {train_dicts[0]}")
    logger.info(f"Caption Ids: {mura_dataset.train_val_caption_ids}")


    # print first 10 caption
    for i in random.sample(range(len(train_dicts)), 10):
        logger.info(f"Sample {i} caption: {train_dicts[i]['caption']}")

    # for the first split, count occurences of label and Label and plot them in two different bar plots
    train_dicts, _, val_dicts, _ = next(mura_dataset.get_cv_splits())
    train_labels = [d["label"] for d in train_dicts]
    val_labels = [d["label"] for d in val_dicts]
    test_labels = [d["label"] for d in mura_dataset.get_test_dicts()]
    train_anatomy_sites = [d["anatomy_site"] for d in train_dicts]
    val_anatomy_sites = [d["anatomy_site"] for d in val_dicts]
    test_anatomy_sites = [d["anatomy_site"] for d in mura_dataset.get_test_dicts()]

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    train_anatomy_site_counts = pd.Series(train_anatomy_sites).value_counts()
    val_anatomy_site_counts = pd.Series(val_anatomy_sites).value_counts()
    test_anatomy_site_counts = pd.Series(test_anatomy_sites).value_counts()
    train_anatomy_site_ratios = train_anatomy_site_counts / train_anatomy_site_counts.sum()
    val_anatomy_site_ratios = val_anatomy_site_counts / val_anatomy_site_counts.sum()
    test_anatomy_site_ratios = test_anatomy_site_counts / test_anatomy_site_counts.sum()

    combined_df = pd.DataFrame({
        "anatomy_site": train_anatomy_site_counts.index,
        "train_count": train_anatomy_site_counts.values,
        "val_count": val_anatomy_site_counts.reindex(train_anatomy_site_counts.index, fill_value=0).values,
        "test_count": test_anatomy_site_counts.reindex(train_anatomy_site_counts.index, fill_value=0).values,
        "train_ratio": train_anatomy_site_ratios.reindex(train_anatomy_site_counts.index, fill_value=0).values,
        "val_ratio": val_anatomy_site_ratios.reindex(train_anatomy_site_counts.index, fill_value=0).values,
        "test_ratio": test_anatomy_site_ratios.reindex(train_anatomy_site_counts.index, fill_value=0).values,
    })

    ax = combined_df.plot(kind='bar', x='anatomy_site', y=['train_ratio', 'val_ratio', 'test_ratio'], figsize=(12, 6), rot=45)

    # Add counts on top of the bars
    for i, bar_group in enumerate(ax.containers):
        for j, bar in enumerate(bar_group):
            height = bar.get_height()
            if height == 0:
                continue
            # Get the correct count from the DataFrame
            col_name = ['train_count', 'val_count', 'test_count'][i]
            count = combined_df.iloc[j][col_name]
            col_name_ratio = ['train_ratio', 'val_ratio', 'test_ratio'][i]
            ratio = combined_df.iloc[j][col_name_ratio]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{int(count)}\n({ratio:.2f})',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Final touches
    plt.title("MURA: Anatomy Site Distribution per Dataset (Ratios with Counts) - First CV Split")
    plt.ylabel("Ratio")
    plt.xlabel("Anatomy Site")
    plt.legend(title="Dataset")
    plt.tight_layout()
    # create folder if it does not exist
    os.makedirs("visualizations/data/pretrain/MURA", exist_ok=True)
    plt.savefig("visualizations/data/pretrain/MURA/MURA_anatomy_site_distribution.png")
    logger.info("MURADataset: Anatomy site distribution plot saved to visualitations/data/pretrain/MURA/MURA_anatomy_site_distribution.png")


    train_label_counts = pd.Series(train_labels).value_counts()
    val_label_counts = pd.Series(val_labels).value_counts()
    test_label_counts = pd.Series(test_labels).value_counts()
    train_label_ratios = train_label_counts / train_label_counts.sum()
    val_label_ratios = val_label_counts / val_label_counts.sum()
    test_label_ratios = test_label_counts / test_label_counts.sum()

    combined_df = pd.DataFrame({
        "label": train_label_counts.index,
        "train_count": train_label_counts.values,
        "val_count": val_label_counts.reindex(train_label_counts.index, fill_value=0).values,
        "test_count": test_label_counts.reindex(train_label_counts.index, fill_value=0).values,
        "train_ratio": train_label_ratios.reindex(train_label_counts.index, fill_value=0).values,
        "val_ratio": val_label_ratios.reindex(train_label_counts.index, fill_value=0).values,
        "test_ratio": test_label_ratios.reindex(train_label_counts.index, fill_value=0).values,
    })

    ax = combined_df.plot(kind='bar', x='label', y=['train_ratio', 'val_ratio', 'test_ratio'], figsize=(12, 6), rot=45)

    # Add counts on top of the bars
    for i, bar_group in enumerate(ax.containers):
        for j, bar in enumerate(bar_group):
            height = bar.get_height()
            if height == 0:
                continue
            # Get the correct count from the DataFrame
            col_name = ['train_count', 'val_count', 'test_count'][i]
            count = combined_df.iloc[j][col_name]
            col_name_ratio = ['train_ratio', 'val_ratio', 'test_ratio'][i]
            ratio = combined_df.iloc[j][col_name_ratio]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{int(count)}\n({ratio:.2f})',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Final touches
    plt.title("MURA: Label Distribution per Dataset (Ratios with Counts) - First CV Split")
    plt.ylabel("Ratio")
    plt.xlabel("Label")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig("visualizations/data/pretrain/MURA/MURA_label_distribution.png")
    logger.info("MURADataset: Label distribution plot saved to visualitations/data/pretrain/MURA/MURA_label_distribution.png")