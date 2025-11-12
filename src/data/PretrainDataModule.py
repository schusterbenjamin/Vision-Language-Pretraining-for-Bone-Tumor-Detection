from collections import Counter
import json
import math
import sys
import os
import logging
from typing import Generator, Tuple
import lightning as L
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from overrides import override
import pandas as pd
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    EnsureChannelFirstd,
    Lambdad,
    RandAffined,
    RandRotated,
    RandFlipd,
    RandZoomd,
    HistogramNormalized,
    RandGaussianNoised, 
)
from torchvision.transforms import Grayscale

from tqdm import tqdm
from transformers import DistilBertTokenizer, BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.sampler.UniqueCaptionSampler import NoDuplicateCaptionSampler
from src.data.helpers.hash_list_of_dicts import hash_list_of_strings
from src.data.LERADataset import LERADataset
from src.data.MURADataset import MURADataset
from src.data.transform.PadToSquaredEdgeAverage import PadToSquaredEdgeAverage
from src.data.transform.CropLargerDimension import CropLargerDimension
from src.data.KFoldCVDataModule import DataModuleFolds, KFoldCVDataModule

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class PretrainDataModule(KFoldCVDataModule):
    """
    A PyTorch Lightning data module for pretraining on combined LERA and MURA datasets.
    This data module handles the loading, preprocessing, and cross-validation splitting of 
    medical X-ray images from both LERA and MURA datasets for pretraining vision-language models.
    It combines both datasets and provides K-fold cross-validation splits with proper data 
    augmentations and normalization.

    It is not a standard DataModule, but yields DataModuleFolds for each fold in K-Fold CV.

    Args
    ----------
        captions_path (str, optional): Path to the CSV file containing image captions. 
            Defaults to "/mnt/nfs/homedirs/benjamins/project/res/data/pretrain/captions.csv".
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        num_channels (int, optional): Number of output channels (1 for grayscale, 3 for RGB). 
            Must be 1 or 3. Defaults to 3.
        tokenizer (str, optional): Tokenizer type to use ('distilbert' or 'tinybert'). 
            Defaults to 'distilbert'.
        try_with_only_n_samples (int, optional): If specified, limits the number of samples 
            used for experimentation. Defaults to None.
        disable_augmentations (bool, optional): If True, disables all data augmentations. 
            Defaults to False.
    
    Methods
    ----------
        get_cv_splits(): Generator yielding (DataModuleFolds, label_weights (not used)) for each fold.

    Notes:
        Environment Variables:
            MURA_DATASET_PATH: Path to the MURA dataset directory
            LERA_DATASET_PATH: Path to the LERA dataset directory
        Note:
            - Images are preprocessed with histogram normalization, cropping, padding, and resizing
            - Augmentations include random affine transforms, rotation, flipping, zooming, and Gaussian noise
            - Captions are tokenized and padded to a maximum length of 40 tokens
            - Mean and std statistics are computed and cached for each CV fold
            - Uses NoDuplicateCaptionSampler to avoid duplicate captions in batches
    """
    def __init__(
        self,
        captions_path="/mnt/nfs/homedirs/benjamins/project/res/data/pretrain/captions.csv",
        batch_size=32,
        num_workers=2,
        num_channels=3,
        tokenizer='distilbert',
        try_with_only_n_samples=None,
        disable_augmentations=False,
        #TODO: think about whether to make Gaussian noise augmentation optional or not
    ):
        super().__init__()

        try:
            mura_path = os.environ.get("MURA_DATASET_PATH")
            lera_path = os.environ.get("LERA_DATASET_PATH")
        except Exception as e:
            logger.error(
                "PretrainDataModule: Please set the MURA_DATASET_PATH and LERA_DATASET_PATH environment variables to the paths of the MURA and LERA datasets."
            )
            raise e

        if not (num_channels == 1 or num_channels == 3):
            raise ValueError(
                f"PretrainDataModule: num_channels must be 1 or 3, but got {num_channels}"
            )
        if tokenizer=='distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        elif tokenizer == "tinybert":
            tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        
        self.tokenizer = tokenizer
        self.mura_path = mura_path
        self.lera_path = lera_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_channels = num_channels
        self.try_with_only_n_samples = try_with_only_n_samples

        logger.info("PretrainDataModule: Initializing the LERA dataset")
        self.lera_dataset = LERADataset(
            path=self.lera_path,
            captions_path=captions_path,
        )
        logger.info("PretrainDataModule: Initializing the MURA dataset")
        self.mura_dataset = MURADataset(
            path=self.mura_path,
            captions_path=captions_path,
        )
        logger.info("PretrainDataModule: Tokenizing captions for LERA and MURA datasets.")
        # I stood in front of the decision to handle tokenization of the captions in the LERA and MURA datasets separately or together.
        # I decided to handle them together, since this way I can ensure that the tokens have the same length and are padded to the same length.
        all_dicts = self.lera_dataset.train_val_dicts + self.lera_dataset.test_dicts + self.mura_dataset.train_val_dicts + self.mura_dataset.test_dicts
        captions = [d["caption"] for d in all_dicts]
        
        tokenized_captions = self.tokenize_captions(captions)

        for i, d in enumerate(all_dicts):
            d['caption_tokenized'] = {}
            for key, value in tokenized_captions.items():
                d['caption_tokenized'][key] = value[i]
        
        # logger.debug(f"PretrainDataModule: first train dict from LERA dataset: {self.lera_dataset.train_val_dicts[0]}")

        logger.info("PretrainDataModule: Finished setting up the datasets")

        image_size = (224, 224)  # default image size for most pretrained models

        pytorch_grayscale_transform = Grayscale(num_output_channels=1)
        transforms_before_normalization_list = [
            LoadImaged(keys=["x-ray"]),
            # add the channel dimension to the image
            EnsureChannelFirstd(keys=["x-ray"]),
            # Most samples already are grayscale, but I found one that has 3 channels, so I need this transform
            Lambdad(keys=["x-ray"], func=pytorch_grayscale_transform),
            # perform histogram equalization (part of mission STOSOP)
            HistogramNormalized(keys=["x-ray"]),
            # e.g. ViT requires 3 channels, so depending on the num_channels, we need to expand to 3 channels again
            (
                Lambdad(keys=["x-ray"], func=lambda x: x.repeat(3, 1, 1))
                if num_channels == 3
                else None
            ),
            # conservatively crop the larger dimension by a maximum of 5% of the size of the larger dimension
            CropLargerDimension(keys=["x-ray"], maximum_crop_ratio=0.05),
            # pad to square image
            PadToSquaredEdgeAverage(keys=["x-ray"]),
            # resize to 224x224
            Resized(keys=["x-ray"], spatial_size=image_size),
        ]
        transforms_before_normalization_list = [
            t for t in transforms_before_normalization_list if t is not None
        ]
        self.transforms_before_normalization = Compose(
            transforms_before_normalization_list
        )

        self.augmentations = Compose(
            [
                RandAffined(keys=["x-ray"], prob=0.3, translate_range=[20, 20],
                            shear_range=[5, 5],
                            mode=["bilinear"], padding_mode="border"),
                RandRotated(keys=["x-ray"], prob=0.3, range_x=(math.pi / 6)), # pi / 6 = 30 degrees, mentioned in Michaels master thesis
                RandFlipd(keys=["x-ray"], prob=0.3, spatial_axis=0),
                RandZoomd(keys=["x-ray"], prob=0.3, min_zoom=1.1, max_zoom=1.3),
                # Lambdad(keys=["x-ray"], func=RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))),
                RandGaussianNoised(keys=["x-ray"], prob=0.5, mean=0.0, std=0.01),
                # Lambdad(keys=["x-ray"], func=ColorJitter(brightness=0.2, contrast=0.2)),
            ]
        )
        if disable_augmentations:
            self.augmentations = Compose()

        if len(self.augmentations.transforms) == 0:
            logger.warning(
                "PretrainDataModule: No augmentations are applied."
            )

        # this is just a placeholder for the mean and std for each fold
        self.mean, self.std = [None] * 5, [None] * 5

    def tokenize_captions(self, captions):
        # tokenize the captions
        tokenized_captions = self.tokenizer(
            captions, padding=True, truncation=True, max_length=40, return_tensors="pt"
        )
        return tokenized_captions

    def _get_mean_and_std(self, data_dicts):
        # The computation of mean and std is quite computationally expensive, so the result will be store in a json in datacache/<datadict_hash>.json
        # So, if the mean and std are already cached, we can just load them from there
        image_paths = [item["x-ray"] for item in data_dicts]
        data_dicts_hash = hash_list_of_strings(image_paths)
        cache_path = os.path.join("datacache/", f"{data_dicts_hash}.json")
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    mean = data["mean"]
                    std = data["std"]
                    logger.info(
                        f"PretrainDataModule: Loaded mean and std from cache at {cache_path}"
                    )
                    return mean, std
        except Exception as e:
            logger.info(
                "PretrainDataModule: Tried but failed to load mean and std from cache. Will compute them instead."
            )

        # Store all images in a list
        image_list = []

        for item in data_dicts:
            loaded = self.transforms_before_normalization(item)
            img = loaded["x-ray"]
            image_list.append(img)

        # Stack into a single NumPy array
        all_images = np.stack(
            [img.numpy() for img in image_list]
        )  # shape: (N, C, H, W)

        # Compute mean and std across the dataset
        mean = all_images.mean()
        std = all_images.std()

        mean, std = float(mean), float(std)

        logger.debug(
            f"PretrainDataModule: Mean and std computed over {len(all_images)} images with shape {all_images.shape}"
        )
        # Save mean and std to cache
        with open(cache_path, "w") as f:
            json.dump({"mean": mean, "std": std}, f)
            logger.debug(
                f"PretrainDataModule: Saved mean and std to cache at {cache_path}"
            )

        return mean, std

    @override
    def get_cv_splits(self) -> Generator[Tuple[L.LightningDataModule, Tuple[float, float]], None, None]:
        """
        Generate cross-validation splits with combined training data and separate validation sets.
        Yields:
            Tuple:
                - DataModuleFolds: Lightning data module with train and validation dataloaders
                - Tuple[float, float]: not needed just here to fulfill the interface
        """
        for i, ((lera_train_data, lera_train_caption_ids, lera_val_data, lera_val_caption_ids), (mura_train_data, mura_train_caption_ids, mura_val_data, mura_val_caption_ids)) in enumerate(zip(self.lera_dataset.get_cv_splits(), self.mura_dataset.get_cv_splits())):
            logger.info(f"PretrainDataModule: Generating K-Fold Cross Validation Split Number {i}")
            train_combined = lera_train_data + mura_train_data
            train_caption_ids_combined = lera_train_caption_ids + mura_train_caption_ids

            # for each split, compute the mean and std of the training data for normalization
            self.mean[i], self.std[i] = self._get_mean_and_std(train_combined)
            transforms_without_augmentations = Compose(
                [
                    self.transforms_before_normalization,
                    NormalizeIntensityd(keys=["x-ray"], subtrahend=self.mean[i], divisor=self.std[i])
                ]
            )
            transforms_with_augmentations = Compose(
                [
                    transforms_without_augmentations,
                    self.augmentations,
                ]
            )

            if self.try_with_only_n_samples is not None:
                # if the user wants to try with only n samples, we need to sample the data
                indices = np.random.choice(len(train_combined), size=self.try_with_only_n_samples, replace=False)
                # Apply the same indices to both lists
                train_combined = [train_combined[i] for i in indices]
                train_caption_ids_combined = [train_caption_ids_combined[i] for i in indices]

                
                indices = np.random.choice(len(lera_val_data), size=min(self.try_with_only_n_samples, len(lera_val_data)), replace=False)
                lera_val_data = [lera_val_data[i] for i in indices]
                lera_val_caption_ids = [lera_val_caption_ids[i] for i in indices]
                
                indices = np.random.choice(len(mura_val_data), size=min(self.try_with_only_n_samples, len(mura_val_data)), replace=False)
                mura_val_data = [mura_val_data[i] for i in indices]
                mura_val_caption_ids = [mura_val_caption_ids[i] for i in indices]

            train_dataset = Dataset(data=train_combined, transform=transforms_with_augmentations)
            lera_val_dataset = Dataset(data=lera_val_data, transform=transforms_without_augmentations)
            mura_val_dataset = Dataset(data=mura_val_data, transform=transforms_without_augmentations)

            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=NoDuplicateCaptionSampler(
                    dataset=train_dataset, batch_size=self.batch_size, caption_ids=train_caption_ids_combined, probabilistic_mode="full"),
                # batch_size=self.batch_size,
                # shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            lera_val_dataloader = DataLoader(
                lera_val_dataset,
                batch_sampler=NoDuplicateCaptionSampler(
                    dataset=lera_val_dataset, batch_size=self.batch_size, caption_ids=lera_val_caption_ids, probabilistic_mode="semi", deterministic=True),
                # batch_size=self.batch_size,
                # shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            mura_val_dataloader = DataLoader(
                mura_val_dataset,
                batch_sampler=NoDuplicateCaptionSampler(
                    dataset=mura_val_dataset, batch_size=self.batch_size, caption_ids=mura_val_caption_ids, probabilistic_mode="semi", deterministic=True),
                # batch_size=self.batch_size,
                # shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            yield DataModuleFolds(
                train_dataloader=train_dataloader,
                val_dataloaders=[lera_val_dataloader, mura_val_dataloader],
            ), [0.0, 0.0]

    @override
    def test_dataloader(self) -> DataLoader:
        raise Exception(
            "Are you sure you want to use the test dataloader??? You should only use this in the very very end for one time evaluation. Once used, there is no going back! If you are sure, comment out this line that raises the exception."
        )
        # return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # lera_test_dataloader = DataLoader(self.lera_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # mura_test_dataloader = DataLoader(self.mura_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # return [lera_test_dataloader, mura_test_dataloader] # NOTE: DO NOT CHANGE THE ORDER OF THE DATALOADERS! The first dataloader is for the LERA dataset and the second dataloader for the MURA dataset


if __name__ == "__main__":
    # please only enable one of the following flags at a time, since the get_cv_splits() method is a generator and will yield only once
    stratification_over_anatomy_site = False
    image_examples = False
    augmentation_examples = False
    unique_caption_occurences = False
    print_unique_anatomy_sites = False
    print_first_batch = False
    calculate_average_caption_duplication_per_batch = False
    save_all_metadata_as_csv = True
    assert save_all_metadata_as_csv + stratification_over_anatomy_site + image_examples + augmentation_examples + unique_caption_occurences + print_unique_anatomy_sites + print_first_batch + calculate_average_caption_duplication_per_batch == 1, "Please enable only one of the flags at a time."

    L.seed_everything(42)  # Set seed for reproducibility
    # Test the DataModule
    pretrain_data_module = PretrainDataModule(batch_size=256)

    ## force caching of the datasets
    # datamodule, _ = next(downstream_data_module.get_cv_splits())
    # train_loader = datamodule.train_dataloader()
    # val_loaders = datamodule.val_dataloader()
    # lera_loader = val_loaders[0]
    # mura_loader = val_loaders[1]

    # for batch in train_loader:
    #     _ = batch["x-ray"]
    # for batch in lera_loader:
    #     _ = batch["x-ray"]
    # for batch in mura_loader:
    #     _ = batch["x-ray"]

    if calculate_average_caption_duplication_per_batch:
        datamodule, _ = next(pretrain_data_module.get_cv_splits())
        train_loader = datamodule.train_dataloader()
        # Initialize list to store percentages
        duplicate_caption_percentages = []

        # Iterate through all batches
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches")):
            # Get captions from the batch
            captions = batch["caption"]

            # Count occurrences of each caption
            caption_counts = Counter(captions)
            
            # Count entries with duplicate captions directly from caption_counts
            entries_with_duplicate_captions = sum(count for caption, count in caption_counts.items() if count > 1)
            
            # Calculate percentage
            percentage = (entries_with_duplicate_captions / len(captions)) * 100
            duplicate_caption_percentages.append(percentage)

        # Calculate statistics
        if duplicate_caption_percentages:
            min_percentage = min(duplicate_caption_percentages)
            max_percentage = max(duplicate_caption_percentages)
            median_percentage = np.median(duplicate_caption_percentages)
            avg_percentage = np.mean(duplicate_caption_percentages)
            
            # Report results
            logger.info(f"Duplicated captions across train dataset")
            logger.info(f"Caption duplication statistics across {len(duplicate_caption_percentages)} batches:")
            logger.info(f"Min percentage: {min_percentage:.2f}%")
            logger.info(f"Max percentage: {max_percentage:.2f}%")
            logger.info(f"Median percentage: {median_percentage:.2f}%")
            logger.info(f"Average percentage: {avg_percentage:.2f}%")
        else:
            logger.warning("No valid batches processed, cannot calculate statistics")

        val_dataloaders = datamodule.val_dataloader()

        duplicate_caption_percentages = []
        for val_loader in val_dataloaders:
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing val batches")):
                # Get captions from the batch
                captions = batch["caption"]

                # Count occurrences of each caption
                caption_counts = Counter(captions)
                
                # Count entries with duplicate captions directly from caption_counts
                entries_with_duplicate_captions = sum(count for caption, count in caption_counts.items() if count > 1)
                
                # Calculate percentage
                percentage = (entries_with_duplicate_captions / len(captions)) * 100
                duplicate_caption_percentages.append(percentage)
        
        # Calculate statistics
        if duplicate_caption_percentages:
            min_percentage = min(duplicate_caption_percentages)
            max_percentage = max(duplicate_caption_percentages)
            median_percentage = np.median(duplicate_caption_percentages)
            avg_percentage = np.mean(duplicate_caption_percentages)
            
            # Report results
            logger.info(f"Duplicated captions across val dataset")
            logger.info(f"Caption duplication statistics across {len(duplicate_caption_percentages)} batches:")
            logger.info(f"Min percentage: {min_percentage:.2f}%")
            logger.info(f"Max percentage: {max_percentage:.2f}%")
            logger.info(f"Median percentage: {median_percentage:.2f}%")
            logger.info(f"Average percentage: {avg_percentage:.2f}%")
            
        
        

    if print_first_batch:
        datamodule, _ = next(pretrain_data_module.get_cv_splits())
        train_loader = datamodule.train_dataloader()
        first_train_batch = next(iter(train_loader))
        logger.info(f"PretrainDataModule: First train batch {first_train_batch}")
        val_loader = datamodule.val_dataloader()[0]  # Get the first validation loader (LERA)
        first_val_batch = next(iter(val_loader))
        logger.info(f"PretrainDataModule: First validation batch {first_val_batch}")

    if print_unique_anatomy_sites:
        datamodule, _ = next(pretrain_data_module.get_cv_splits())
        train_loader = datamodule.train_dataloader()
        val_loaders = datamodule.val_dataloader()
        anatomy_sites = []
        for batch in train_loader:
            anatomy_sites.extend(batch['anatomy_site'])
        for val_loader in val_loaders:
            for batch in val_loader:
                anatomy_sites.extend(batch['anatomy_site'])
        unique_anatomy_sites = set(anatomy_sites)
        logger.info(f"PretrainDataModule: Unique anatomy sites in the dataset: {unique_anatomy_sites}")


    if unique_caption_occurences:
        datamodule, _ = next(pretrain_data_module.get_cv_splits())
        train_loader = datamodule.train_dataloader()
        val_loaders = datamodule.val_dataloader()
        anatomy_sites = []
        for batch in train_loader:
            anatomy_sites.extend(batch['caption'])
        for val_loader in val_loaders:
            for batch in val_loader:
                anatomy_sites.extend(batch['caption'])
        caption_counts = Counter(anatomy_sites)
        # store the unique captions and their counts in a DataFrame
        df = pd.DataFrame(caption_counts.items(), columns=['caption', 'count'])
        df = df.sort_values(by='count', ascending=False)
        # save the DataFrame to a csv file
        df.to_csv("visualizations/data/pretrain/unique_captions_train_val.csv")



    if stratification_over_anatomy_site:
        # Calculate and plot the ratios of anatomy sites across the test and train/val splits
        # NOTE: remove transformations above to speed up the process, the images are not needed for this calculation
        records = []
        for i, (fold_data_module, _) in enumerate(pretrain_data_module.get_cv_splits()):
            logger.debug(f"PretrainDataModule: K-Fold Cross Validation Split: {i}")
            
            train_loader = fold_data_module.train_dataloader()
            val_loaders = fold_data_module.val_dataloader()
            lera_val_loader, mura_val_loader = val_loaders[0], val_loaders[1]

            for j, train_batch in enumerate(train_loader):
                anatomy_sites = train_batch["anatomy_site"]
                labels = train_batch["label"].tolist()
                train_key = f"train_{i}"
                batch_records = list(zip(anatomy_sites, [train_key] * len(anatomy_sites), labels))
                records.extend(batch_records)
            logger.debug(f"PretrainDataModule: Added anatomy sites from train for split {i}")

            for j, val_batch in enumerate(lera_val_loader):
                anatomy_sites = val_batch["anatomy_site"]
                labels = val_batch["label"].tolist()
                val_key = f"val_{i}"
                batch_records = list(zip(anatomy_sites, [val_key] * len(anatomy_sites), labels))
                records.extend(batch_records)
            for j, val_batch in enumerate(mura_val_loader):
                anatomy_sites = val_batch["anatomy_site"]
                labels = val_batch["label"].tolist()
                val_key = f"val_{i}"
                batch_records = list(zip(anatomy_sites, [val_key] * len(anatomy_sites), labels))
                records.extend(batch_records)
            logger.debug(f"PretrainDataModule: Added anatomy sites from for split {i}")

        lera_test_dicts = pretrain_data_module.lera_dataset.get_test_dicts()
        lera_test_anatomy_sites = [item["anatomy_site"] for item in lera_test_dicts]
        lera_test_labels = [item["label"] for item in lera_test_dicts]
        lera_test_records = list(zip(lera_test_anatomy_sites, ["test"] * len(lera_test_anatomy_sites), lera_test_labels))
        records.extend(lera_test_records)
        mura_test_dicts = pretrain_data_module.mura_dataset.get_test_dicts()
        mura_test_anatomy_sites = [item["anatomy_site"] for item in mura_test_dicts]
        mura_test_labels = [item["label"] for item in mura_test_dicts]
        mura_test_records = list(zip(mura_test_anatomy_sites, ["test"] * len(mura_test_anatomy_sites), mura_test_labels))
        records.extend(mura_test_records)

        df = pd.DataFrame(records, columns=["anatomy_site", "dataset", "label"])
        counts = df.groupby(['anatomy_site', 'dataset']).size().reset_index(name='count')
        total_per_dataset = counts.groupby('dataset')['count'].transform('sum')
        counts['ratio'] = counts['count'] / total_per_dataset
        
        plt.figure(figsize=(14, 6))
        # sort it by the ratio
        counts = counts.sort_values(by='ratio', ascending=False)
        # ensure the order of datasets is train_0, train_1, train_2, train_3, val_0, val_1, val_2, val_3, test
        counts['dataset'] = pd.Categorical(counts['dataset'], categories=['train_0', 'train_1', 'train_2', 'train_3', 'val_0', 'val_1', 'val_2', 'val_3', 'test'], ordered=True)
        sns.barplot(data=counts, 
                    x="anatomy_site", 
                    y="ratio", 
                    hue="dataset")
        plt.xticks(rotation=45, ha='right')
        plt.title("Anatomy Site Distribution per Dataset")
        plt.ylabel("Proportion (Ratio)")
        plt.xlabel("Anatomy Site")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig("visualizations/data/pretrain/stratification_over_anatomy_site.png", dpi=300)


        # Calculate pertentage of occurence of combination of anatomy site and label over train and val
        # For that we take the first split
        records_train_val = df[(df['dataset'] == 'train_0') | (df['dataset'] == 'val_0')]
        counts = records_train_val.groupby(['anatomy_site', 'label']).size().reset_index(name='count')
        counts['percentage'] = counts['count'] / counts['count'].sum()

        # Plot percentages of anatomy site and label combinations
        plt.figure(figsize=(14, 6))
        counts_sorted = counts.sort_values(by='percentage', ascending=False)
        sns.barplot(data=counts_sorted, x='anatomy_site', y='percentage', hue='label')
        plt.xticks(rotation=45, ha='right')
        plt.title("Percentage by Anatomy Site and Label on train+val set")
        plt.ylabel("Percentage")
        plt.xlabel("Anatomy Site")
        plt.legend(title="Label (0=Normal, 1=Abnormal)")
        plt.tight_layout()
        plt.savefig("visualizations/data/pretrain/percentage_by_anatomy_site_and_label_on_train_and_val.png", dpi=300)

        # Save counts to CSV
        counts.to_csv("visualizations/data/pretrain/counts_by_anatomy_site_and_label_on_train_and_val.csv", index=False)

        
    
    if image_examples or augmentation_examples:
        fold_data_module, _ = next(pretrain_data_module.get_cv_splits())
        train_loader = fold_data_module.train_dataloader()
        train_batch = next(iter(train_loader))

        # get the first n images from each dataset
        n = 5
        lera_examples = []
        i = 0
        while len(lera_examples) < n:
            sample_dataset = train_batch["dataset"][i]
            if sample_dataset == "LERA":
                lera_examples.append({
                    "x-ray": train_batch["x-ray"][i],
                    "image_path": train_batch["image_path"][i],
                    "label": train_batch["label"][i],
                    "anatomy_site": train_batch["anatomy_site"][i],
                    "dataset": train_batch["dataset"][i]
                })
            i += 1
        logger.debug(f"PretrainDataModule: Found {len(lera_examples)} LERA examples")
        mura_examples = []
        i = 0
        while len(mura_examples) < n:
            sample_dataset = train_batch["dataset"][i]
            if sample_dataset == "MURA":
                mura_examples.append({
                    "x-ray": train_batch["x-ray"][i],
                    "image_path": train_batch["image_path"][i],
                    "label": train_batch["label"][i],
                    "anatomy_site": train_batch["anatomy_site"][i],
                    "dataset": train_batch["dataset"][i]
                })
            i += 1
        logger.debug(f"PretrainDataModule: Found {len(mura_examples)} MURA examples")


    if image_examples:
        # Plot the images in lera/mura_examples
        plt.figure(figsize=(15, 5))
        for i, example in enumerate(lera_examples):
            # First column: original image
            ax1 = plt.subplot(2, n * 2, i * 2 + 1)
            img = Image.open(example["image_path"])
            ax1.imshow(img, cmap="gray")
            ax1.axis("off")

            # Second column: transformed image
            ax2 = plt.subplot(2, n * 2, i * 2 + 2)
            ax2.imshow(example["x-ray"][0].numpy().T, cmap="gray")
            ax2.axis("off")
            

            # Shared title across both images
            title = (
                f"LERA\nLabel: {example['label']}\n"
                f"Anatomy Site: {example['anatomy_site']}"
            )
            ax1.set_title(title, fontsize=8)

        for i, example in enumerate(mura_examples):
            ax1 = plt.subplot(2, n * 2, n * 2 + i * 2 + 1)
            img = Image.open(example["image_path"])
            ax1.imshow(img, cmap="gray")
            ax1.axis("off")

            ax2 = plt.subplot(2, n * 2, n * 2 + i * 2 + 2)
            ax2.imshow(example["x-ray"][0].numpy().T, cmap="gray")
            ax2.axis("off")
            

            title = (
                f"MURA\nLabel: {example['label']}\n"
                f"Anatomy Site: {example['anatomy_site']}"
            )
            ax1.set_title(title, fontsize=8)

        plt.tight_layout()
        plt.savefig("visualizations/data/pretrain/image_examples_with_histogram_equalization.png", dpi=300)

    if augmentation_examples:
        # NOTE: in order for this to work, make sure that the augmentations in the PretrainDataModule are not applied to the dataset (e.g. comment them out)
        # NOTE: also, make sure that the augmentations here match the ones in the PretrainDataModule

        augmentations = Compose(
            [
                RandAffined(keys=["x-ray"], prob=0.3, translate_range=[20, 20], mode=["bilinear"], padding_mode="border"),
                RandRotated(keys=["x-ray"], prob=0.3, range_x=(math.pi / 6)), # pi / 6 = 30 degrees, mentioned in Michaels master thesis
                RandFlipd(keys=["x-ray"], prob=0.3, spatial_axis=0),
                RandZoomd(keys=["x-ray"], prob=0.3, min_zoom=1.1, max_zoom=1.3),
                RandGaussianNoised(keys=["x-ray"], prob=0.5, mean=0.0, std=0.01)
            ]
        )

        # for each sample apply the augmentations 4 times
        lera_examples_augmented = []
        for lera_sample in lera_examples:
            lera_sample_augmented = []
            for i in range(4):
                lera_sample_augmented.append(augmentations(lera_sample))
            lera_examples_augmented.append(lera_sample_augmented)
        mura_examples_augmented = []
        for mura_sample in mura_examples:
            mura_sample_augmented = []
            for i in range(4):
                mura_sample_augmented.append(augmentations(mura_sample))
            mura_examples_augmented.append(mura_sample_augmented)

        def plot_augmented_samples(original_samples, augmented_samples, title_prefix):
            n_samples = len(original_samples)
            n_augmentations = len(augmented_samples[0])  # Assuming 4 augmentations

            fig, axes = plt.subplots(
                n_samples, n_augmentations + 1, figsize=(15, 3 * n_samples)
            )
            fig.suptitle(f"{title_prefix} Samples with Augmentations", fontsize=16)

            for row in range(n_samples):
                sample = original_samples[row]
                original_image = sample["x-ray"]
                label = sample["label"]

                # First column: original image
                ax = axes[row, 0] if n_samples > 1 else axes[0]
                ax.imshow(original_image[0], cmap="gray")
                ax.set_title(f"Original\nLabel: {label}")
                ax.axis("off")

                # Augmented images
                for col in range(n_augmentations):
                    augmented_image = augmented_samples[row][col]["x-ray"]
                    ax = axes[row, col + 1] if n_samples > 1 else axes[col + 1]
                    ax.imshow(augmented_image[0], cmap="gray")
                    ax.set_title(f"Aug {col + 1}\nabel: {label}")
                    ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"visualizations/data/pretrain/augmented_samples_{title_prefix}.png", dpi=300)

        plot_augmented_samples(lera_examples, lera_examples_augmented, "LERA")
        plot_augmented_samples(mura_examples, mura_examples_augmented, "MURA")

    if save_all_metadata_as_csv:
        mura_train_val_dicts = pretrain_data_module.mura_dataset.train_val_dicts
        mura_train_val_df = pd.DataFrame(mura_train_val_dicts)
        mura_train_val_df['set'] = 'train/val'
        mura_train_val_df.drop('patient_id', inplace=True, axis=1)
        mura_test = pretrain_data_module.mura_dataset.test_dicts
        mura_test_df = pd.DataFrame(mura_test)
        mura_test_df['set'] = 'test'
        lera_train_val_dicts = pretrain_data_module.lera_dataset.train_val_dicts
        lera_train_val_df = pd.DataFrame(lera_train_val_dicts)
        lera_train_val_df['set'] = 'train/val'
        lera_train_val_df.drop('case_number', inplace=True, axis=1)
        lera_test = pretrain_data_module.lera_dataset.test_dicts
        lera_test_df = pd.DataFrame(lera_test)
        lera_test_df['set'] = 'test'
        
        total_df = pd.concat([mura_train_val_df, mura_test_df, lera_train_val_df, lera_test_df], axis=0)
        total_df.drop(['x-ray', 'image_path', 'caption', 'caption_tokenized'], inplace=True, axis=1)

        total_df.to_csv('visualizations/data/pretrain/metadata.csv')

