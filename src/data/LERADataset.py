import logging
import logging.config
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.helpers.ensure_same_test_set import load_test_and_train_split_from_saved_test_set_info, save_test_set_hash, save_test_set_info

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class LERADataset:
    """
    A dataset class for loading and processing LERA X-ray images with captions.
    This class handles loading X-ray images from the LERA dataset, assigns appropriate
    captions based on anatomy site and abnormality labels, and provides cross-validation
    splits while ensuring no patient data leakage between train, validation and test sets.

    Args
    ----------
        path (str): Path to the LERA dataset directory
        captions_path (str): Path to the captions CSV file

    Attributes
    ----------
        train_val_dicts (list): Training and validation data dictionaries
        test_dicts (list): Test data dictionaries
        caption_ids (list): Corresponding caption IDs for each sample

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

        data_dicts, caption_ids = self._get_data_as_dict()
        self.caption_ids = caption_ids
        self.train_val_dicts, self.test_dicts = self._split_test(data_dicts)


        logger.debug(f"LERADataset: Number of training/validation samples: {len(self.train_val_dicts)}")
        logger.debug(f"LERADataset: Number of test samples: {len(self.test_dicts)}")
        logger.debug(
            f"LERADataset: Ratio train+val/test: {len(self.train_val_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}/{len(self.test_dicts)/len(self.train_val_dicts+self.test_dicts):.2f}"
        )

        # assert that no patient has images in two different sets
        train_patients = set([d["case_number"] for d in self.train_val_dicts])
        test_patients = set([d["case_number"] for d in self.test_dicts])
        assert (
            len(train_patients.intersection(test_patients)) == 0
        ), "There is at least one patient who has images in both train/val and test set"
        # case_number was only needed for a proper split and needs to be removed now, s.t. it has the same structure as the LERA dataset
        # for train_val dicts it gets removed after the split, since it is needed for the split it is not removed here
        for d in self.test_dicts:
            d.pop("case_number")

        logger.info("LERADataset: Successfully initialized the dataset")

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
        Load or create LERA dataset and return it as a list of dictionaries.
        
        This method first attempts to load an existing 'dataset.csv' file from the dataset path.
        If the file doesn't exist, it creates a new dataset by:
        1. Reading the 'labels.csv' file to get case information
        2. Traversing subfolders to find PNG images in 'ST-1' directories
        3. Combining image paths with corresponding anatomy site and label information
        4. Saving the consolidated dataset to 'dataset.csv'
        
        For each image in the dataset, it generates a caption using the anatomy site and label,
        then creates a dictionary containing all relevant information.
        
        Returns:
            tuple: A tuple containing:
                - dicts (list): List of dictionaries, each containing image metadata including:
                    - dataset: Always "LERA"
                    - x-ray: Path to the image file
                    - image_path: Copy of x-ray path for visualization purposes
                    - label: abnormality label
                    - anatomy_site: Anatomical site (with "XR " prefix removed)
                    - caption: Generated caption from anatomy site and label
                    - case_number: Case identifier for dataset splitting
                - caption_ids (list): List of caption indices corresponding to each image
        """
        if os.path.exists(os.path.join(self.path, 'dataset.csv')):
            logger.info(f"LERADataset: Trying to load {os.path.join(self.path, 'dataset.csv')}...")
            lera_images_df = pd.read_csv(os.path.join(self.path, 'dataset.csv'))
            logger.info(f"LERADataset: Loaded existing dataset from {os.path.join(self.path, 'dataset.csv')}")
        else:
            logger.info(f"LERADataset: {os.path.join(self.path, 'dataset.csv')} does not exist, creating a new one. By traversing the dataset folder and reading the labels.csv file.")
            lera_df = pd.read_csv(os.path.join(self.path, 'labels.csv'), header=None)
            lera_images_df = pd.DataFrame(columns=['image_path', 'case_number', 'anatomy_site', 'label'])

            subfolders = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]

            for subfolder in subfolders:
                images = [f for f in os.listdir(os.path.join(os.path.join(self.path, subfolder), 'ST-1')) if f.endswith('.png')]
                if len(images) == 0:
                    print(f'No images found in {os.path.join(self.path, subfolder)}')
                    continue
                for image in images:
                    image_path = os.path.join(os.path.join(self.path, subfolder), 'ST-1', image)
                    anatomy_site = lera_df[lera_df[0] == int(subfolder)].iloc[0, 1]
                    anatomy_site = anatomy_site.replace("XR ", "")
                    label = lera_df[lera_df[0] == int(subfolder)].iloc[0, 2]
                    lera_images_df = pd.concat([lera_images_df, pd.DataFrame({'image_path': [image_path], 'case_number': [int(subfolder)], 'anatomy_site': [anatomy_site], 'label': [label], 'image_id': [os.path.joint(subfolder, 'ST-1', image)]})], ignore_index=True)

            lera_images_df = lera_images_df.reset_index(drop=True)
            lera_images_df.to_csv(os.path.join(self.path, 'dataset.csv'), index=False)

        dicts = []
        caption_ids = []
        for _, row in lera_images_df.iterrows():
            caption, caption_idx = self._get_caption(row["anatomy_site"], row["label"])
            dicts.append(
                {
                    "dataset": "LERA",
                    "x-ray": row["image_path"],
                    "image_path": row["image_path"], # Just added for visualization purposes, to be able to see difference before and after transformations
                    "label": row["label"],
                    # metadata
                    "anatomy_site": row['anatomy_site'],
                    "caption": caption,
                    # case_number is an exception to having the same structure as the MURA dataset.
                    # The pcase_number is only used for splitting and is removed afterwards.
                    "case_number": row['case_number'],
                }
            )
            caption_ids.append(caption_idx)

        return dicts, caption_ids
    
    def _split_test(self, data: list):
        """
        Split the dataset into train/validation and test sets using stratified group k-fold.
        
        Args:
            data (list): List of dictionaries containing dataset samples. Each dictionary should
                        contain 'label', 'anatomy_site', and 'case_number' keys.
        
        Returns:
            tuple: A tuple containing:
                - train_val_samples (list): List of dictionaries for training and validation samples
                - test_samples (list): List of dictionaries for test samples
        
        Raises:
            FileNotFoundError: If no existing test split is found and the method is not modified
                              to allow new split creation.

        Notes:
            This method first attempts to load an existing test split from saved files. If no existing
            split is found, it raises a FileNotFoundError to prevent accidental creation of new splits.
            If the error is bypassed, it creates a new stratified group split based on labels and 
            anatomy sites, ensuring that samples from the same case number stay in the same split.
        """
        train_val_samples, test_samples = load_test_and_train_split_from_saved_test_set_info(dataset_folder=self.path, dataset='LERA', data=data)
        if train_val_samples is None or test_samples is None:
            # to be save throw an error, if the test split is not already created (s.t. there is no way I can accidentally use a wrong test split)
            raise FileNotFoundError(f"LERA: The file with the test set info does not exist. You can comment out this error throw to create a new one, but MAKE SURE THIS IS REALLY WHAT YOU WANT!")
            pass
        else:
            logger.info(f"LERADataset: Using existing test set split.")
            return train_val_samples, test_samples
        logger.warning(f"LERADataset: Creating test set split. ARE YOU SURE THIS IS WHAT YOU WANT? This will overwrite the existing test set split if it exists.")

        stratification_labels = [f"{d['label']}, {d['anatomy_site']}" for d in data]
        groups = [d['case_number'] for d in data]

        # ---- Split: Train/Val (80%) vs Test (20%) ---- #
        sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)

        train_val_indices, test_indices = next(
            sgkf1.split(data, stratification_labels, groups)
        )
        train_val_dicts = [data[i] for i in train_val_indices]
        test_dicts = [data[i] for i in test_indices]

         # save to a csv file the test set split
        save_test_set_info(dataset_folder=self.path, test_samples=test_dicts, train_val_samples=train_val_dicts)
        logger.info(f"LERADataset: Created new test set split and saved it.")

        save_test_set_hash(test_dicts, dataset='LERA')
        logger.info(f"LERADataset: Saved the hash of the test.")

        return train_val_dicts, test_dicts


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
        groups = [d['case_number'] for d in self.train_val_dicts]

        # remove the key value pair "patient_id" from the dicts, since it is only needed for splitting and there needs to be the same keys as in the LERA dataset
        # but do this on a copy, s.t. this function can be called multiple times without side effects
        train_val_dicts = [d.copy() for d in self.train_val_dicts]
        for d in train_val_dicts:
            d.pop("case_number")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        for train_indices, val_indices in sgkf.split(train_val_dicts, stratification_labels, groups):
            train_dicts = [train_val_dicts[i] for i in train_indices]
            val_dicts = [train_val_dicts[i] for i in val_indices]
            train_caption_ids = [self.caption_ids[i] for i in train_indices]
            val_caption_ids = [self.caption_ids[i] for i in val_indices]

            yield train_dicts, train_caption_ids, val_dicts, val_caption_ids

    def get_test_dicts(self):
        """
        Returns the test set dictionaries.

        Returns
        ----------
            test_dicts (list): test data dictionaries"""
        return self.test_dicts


if __name__ == "__main__":
    path = os.environ.get("LERA_DATASET_PATH")
    lera_dataset = LERADataset(path=path)
    len_test_dicts = len(lera_dataset.get_test_dicts())
    for i, (train_dicts, _, val_dicts, _) in enumerate(
        lera_dataset.get_cv_splits()
    ):
        len_train_dicts = len(train_dicts)
        len_val_dicts = len(val_dicts)
        logger.info(f"Fold {i}:")
        logger.info(f"Train dicts: {len_train_dicts}, Validation dicts: {len_val_dicts}, Test dicts: {len_test_dicts}")
        logger.info(f"Ratio train/val/test: {len_train_dicts/(len_train_dicts+len_val_dicts+len_test_dicts):.2f}/{len_val_dicts/(len_train_dicts+len_val_dicts+len_test_dicts):.2f}/{len_test_dicts/(len_train_dicts+len_val_dicts+len_test_dicts):.2f}")

    # # print first sample
    logger.info(f"First sample of train dicts: {train_dicts[0]}")
    logger.info(f"Caption Ids: {lera_dataset.caption_ids}")

    # # for the first split, count occurences of label and Label and plot them in two different bar plots
    # train_dicts, _,  val_dicts, _ = next(lera_dataset.get_cv_splits())
    # train_labels = [d["label"] for d in train_dicts]
    # val_labels = [d["label"] for d in val_dicts]
    # test_labels = [d["label"] for d in lera_dataset.get_test_dicts()]
    # train_anatomy_sites = [d["anatomy_site"] for d in train_dicts]
    # val_anatomy_sites = [d["anatomy_site"] for d in val_dicts]
    # test_anatomy_sites = [d["anatomy_site"] for d in lera_dataset.get_test_dicts()]

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd

    # train_anatomy_site_counts = pd.Series(train_anatomy_sites).value_counts()
    # val_anatomy_site_counts = pd.Series(val_anatomy_sites).value_counts()
    # test_anatomy_site_counts = pd.Series(test_anatomy_sites).value_counts()
    # train_anatomy_site_ratios = train_anatomy_site_counts / train_anatomy_site_counts.sum()
    # val_anatomy_site_ratios = val_anatomy_site_counts / val_anatomy_site_counts.sum()
    # test_anatomy_site_ratios = test_anatomy_site_counts / test_anatomy_site_counts.sum()

    # combined_df = pd.DataFrame({
    #     "anatomy_site": train_anatomy_site_counts.index,
    #     "train_count": train_anatomy_site_counts.values,
    #     "val_count": val_anatomy_site_counts.reindex(train_anatomy_site_counts.index, fill_value=0).values,
    #     "test_count": test_anatomy_site_counts.reindex(train_anatomy_site_counts.index, fill_value=0).values,
    #     "train_ratio": train_anatomy_site_ratios.reindex(train_anatomy_site_counts.index, fill_value=0).values,
    #     "val_ratio": val_anatomy_site_ratios.reindex(train_anatomy_site_counts.index, fill_value=0).values,
    #     "test_ratio": test_anatomy_site_ratios.reindex(train_anatomy_site_counts.index, fill_value=0).values,
    # })

    # ax = combined_df.plot(kind='bar', x='anatomy_site', y=['train_ratio', 'val_ratio', 'test_ratio'], figsize=(12, 6), rot=45)

    # # Add counts on top of the bars
    # for i, bar_group in enumerate(ax.containers):
    #     for j, bar in enumerate(bar_group):
    #         height = bar.get_height()
    #         if height == 0:
    #             continue
    #         # Get the correct count from the DataFrame
    #         col_name = ['train_count', 'val_count', 'test_count'][i]
    #         count = combined_df.iloc[j][col_name]
    #         col_name_ratio = ['train_ratio', 'val_ratio', 'test_ratio'][i]
    #         ratio = combined_df.iloc[j][col_name_ratio]
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             height + 0.01,
    #             f'{int(count)}\n({ratio:.2f})',
    #             ha='center',
    #             va='bottom',
    #             fontsize=8
    #         )

    # # Final touches
    # plt.title("LERA: Anatomy Site Distribution per Dataset (Ratios with Counts) - First CV Split")
    # plt.ylabel("Ratio")
    # plt.xlabel("Anatomy Site")
    # plt.legend(title="Dataset")
    # plt.tight_layout()
    # # create folder if it does not exist
    # os.makedirs("visualizations/data/pretrain/LERA", exist_ok=True)
    # plt.savefig("visualizations/data/pretrain/LERA/LERA_anatomy_site_distribution.png")
    # logger.info("LERADataset: Anatomy site distribution plot saved to visualitations/data/pretrain/LERA/LERA_anatomy_site_distribution.png")


    # train_label_counts = pd.Series(train_labels).value_counts()
    # val_label_counts = pd.Series(val_labels).value_counts()
    # test_label_counts = pd.Series(test_labels).value_counts()
    # train_label_ratios = train_label_counts / train_label_counts.sum()
    # val_label_ratios = val_label_counts / val_label_counts.sum()
    # test_label_ratios = test_label_counts / test_label_counts.sum()

    # combined_df = pd.DataFrame({
    #     "label": train_label_counts.index,
    #     "train_count": train_label_counts.values,
    #     "val_count": val_label_counts.reindex(train_label_counts.index, fill_value=0).values,
    #     "test_count": test_label_counts.reindex(train_label_counts.index, fill_value=0).values,
    #     "train_ratio": train_label_ratios.reindex(train_label_counts.index, fill_value=0).values,
    #     "val_ratio": val_label_ratios.reindex(train_label_counts.index, fill_value=0).values,
    #     "test_ratio": test_label_ratios.reindex(train_label_counts.index, fill_value=0).values,
    # })

    # ax = combined_df.plot(kind='bar', x='label', y=['train_ratio', 'val_ratio', 'test_ratio'], figsize=(12, 6), rot=45)

    # # Add counts on top of the bars
    # for i, bar_group in enumerate(ax.containers):
    #     for j, bar in enumerate(bar_group):
    #         height = bar.get_height()
    #         if height == 0:
    #             continue
    #         # Get the correct count from the DataFrame
    #         col_name = ['train_count', 'val_count', 'test_count'][i]
    #         count = combined_df.iloc[j][col_name]
    #         col_name_ratio = ['train_ratio', 'val_ratio', 'test_ratio'][i]
    #         ratio = combined_df.iloc[j][col_name_ratio]
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             height + 0.01,
    #             f'{int(count)}\n({ratio:.2f})',
    #             ha='center',
    #             va='bottom',
    #             fontsize=8
    #         )

    # # Final touches
    # plt.title("LERA: Label Distribution per Dataset (Ratios with Counts) - First CV Split")
    # plt.ylabel("Ratio")
    # plt.xlabel("Label")
    # plt.legend(title="Dataset")
    # plt.tight_layout()
    # plt.savefig("visualizations/data/pretrain/LERA/LERA_label_distribution.png")
    # logger.info("LERADataset: Label distribution plot saved to visualitations/data/pretrain/LERA/LERA_label_distribution.png")