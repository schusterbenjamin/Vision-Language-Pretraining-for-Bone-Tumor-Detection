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
    ScaleIntensityRanged
)
from torchvision.transforms import Grayscale

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.helpers.hash_list_of_dicts import hash_list_of_strings
from src.data.INTERNALDataset import INTERNALDataset
from src.data.BTXRDDataset import BTXRDDataset
from src.data.transform.PadToSquaredEdgeAverage import PadToSquaredEdgeAverage
from src.data.transform.CropLargerDimension import CropLargerDimension
from src.data.transform.DropChanneld import DropChanneld
from src.data.KFoldCVDataModule import KFoldCVDataModule, DataModuleFolds

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class DownstreamDataModule(KFoldCVDataModule):
    """
    A PyTorch Lightning DataModule for downstream tasks using K-Fold Cross Validation.
    This module combines BTXRD and INTERNAL datasets for binary classification task,
    providing data preprocessing, augmentation, and cross-validation.

    It is not a standard DataModule, but yields DataModuleFolds for each fold in K-Fold CV.
    
    Args
    ----------
        using_crops (bool): Whether to use cropped versions of images from the datasets.
        batch_size (int): Batch size for data loaders. Defaults to 32.
        num_workers (int): Number of worker processes for data loading. Defaults to 2.
        num_channels (int): Number of image channels (1 for grayscale, 3 for RGB). Must be 1 or 3.
        try_with_only_n_samples (int, optional): Limit dataset size to this number of samples for testing.
        gaussian_noise_augmentation (bool): Whether to apply Gaussian noise augmentation. Defaults to True.
        scale_intensity_normalization (bool): If True, uses scale intensity normalization to [-1024, 1024] instead of mean/std normalization. Required for torchxrayvision models. Defaults to False.
        
    Methods
    ----------
        get_cv_splits(): Generator yielding (DataModule, label_weights) for each fold.
        test_dataloader(fold): Returns DataLoader for combined test datasets.
    
    Notes:
        Environment Variables:
            BTXRD_DATASET_PATH: Path to the BTXRD dataset.
            INTERNAL_DATASET_PATH: Path to the INTERNAL dataset.

        Image Processing Pipeline:
            - Load images and ensure channel-first format
            - Drop alpha channel if present and convert to grayscale
            - Apply histogram normalization
            - Expand to 3 channels if required
            - Conservative cropping of larger dimension (max 5%)
            - Pad to square using edge averaging
            - Resize to 224x224
            - Apply normalization (mean/std or scale intensity)
        Augmentations (training only):
            - Random affine transformations (translation, rotation)
            - Random horizontal flipping
            - Random zooming
            - Optional Gaussian noise
        Cross-Validation:
            - Provides K-fold cross-validation splits
            - Computes fold-specific normalization parameters
            - Returns combined training data and separate validation loaders for each dataset
            - Calculates class weights for imbalanced datasets
        Caching:
            - Caches computed mean/std statistics to avoid recomputation
            - Uses MD5 hash of image paths for cache identification
    """

    def __init__(
        self,
        using_crops: bool,
        batch_size=32,
        num_workers=2,
        num_channels=3,
        try_with_only_n_samples=None,
        gaussian_noise_augmentation=True,
        scale_intensity_normalization=False, # use scale intensity normalization to [-1024, 1024] instead of mean/std normalization (needed for models from torchxrayvision)
    ):
        super().__init__()

        try:
            btxrd_path = os.environ.get("BTXRD_DATASET_PATH")
            internal_path = os.environ.get("INTERNAL_DATASET_PATH")
        except Exception as e:
            logger.error(
                "DownstreamDataModule: Please set the BTXRD_DATASET_PATH and INTERNAL_DATASET_PATH environment variables to the paths of the BTXRD and INTERNAL datasets."
            )
            raise e

        if not (num_channels == 1 or num_channels == 3):
            logger.error(
                f"DownstreamDataModule: num_channels must be 1 or 3, but got {num_channels}"
            )
            raise ValueError(
                f"DownstreamDataModule: num_channels must be 1 or 3, but got {num_channels}"
            )

        self.using_crops = using_crops
        self.btxrd_path = btxrd_path
        self.internal_path = internal_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_channels = num_channels
        self.try_with_only_n_samples = try_with_only_n_samples
        self.gaussian_noise_augmentation = gaussian_noise_augmentation
        self.scale_intensity_normalization = scale_intensity_normalization

        logger.info("DownstreamDataModule: Initializing the INTERNAL dataset")
        self.internal_dataset = INTERNALDataset(
            path=self.internal_path,
            using_crops=self.using_crops,
            num_channels=self.num_channels,
        )
        logger.info("DownstreamDataModule: Initializing the BTXRD dataset")
        self.btxrd_dataset = BTXRDDataset(
            path=self.btxrd_path,
            using_crops=self.using_crops,
            num_channels=self.num_channels,
        )
        logger.info("DownstreamDataModule: Finished setting up the datasets")

        pytorch_grayscale_transform = Grayscale(num_output_channels=1)
        transforms_before_normalization_list = [
            LoadImaged(keys=["x-ray"]),
            # add the channel dimension to the image
            EnsureChannelFirstd(keys=["x-ray"]),
            # remove the 4th channel, since it is only 255 throughout the dataset, If there is no 4th channel, this transform does nothing
            DropChanneld(keys=["x-ray"], channel_to_drop=3),
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
            Resized(keys=["x-ray"], spatial_size=(224, 224)),
        ]
        transforms_before_normalization_list = [
            t for t in transforms_before_normalization_list if t is not None
        ]
        self.transforms_before_normalization = Compose(
            transforms_before_normalization_list
        )

        augmentations_list = [
            RandAffined(keys=["x-ray"], prob=0.3, translate_range=[20, 20], mode=["bilinear"], padding_mode="border"),
            RandRotated(keys=["x-ray"], prob=0.3, range_x=(math.pi / 6)), # pi / 6 = 30 degrees, mentioned in Michaels master thesis
            RandFlipd(keys=["x-ray"], prob=0.3, spatial_axis=0),
            RandZoomd(keys=["x-ray"], prob=0.3, min_zoom=1.1, max_zoom=1.3),
            # only include gaussian noise augmentation if it is enabled
            RandGaussianNoised(keys=["x-ray"], prob=0.5, mean=0.0, std=0.01) if self.gaussian_noise_augmentation else None
        ]
        augmentations_list = [t for t in augmentations_list if t is not None]

        self.augmentations = Compose(
            augmentations_list
        )

        if not self.gaussian_noise_augmentation:
            logger.info("DownstreamDataModule: Gaussian noise augmentation is disabled.")

        # this is just a placeholder for the mean and std for each fold
        self.mean, self.std = [None] * 4, [None] * 4


    def _get_mean_and_std(self, data_dicts):
        # The computation of mean and std is quite computationally expensive, so the result will be stored in a json in datacache/<datadict_hash>.json
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
                        f"DownstreamDataModule: Loaded mean and std from cache at {cache_path}"
                    )
                    return mean, std
        except Exception as e:
            logger.info(
                "DownstreamDataModule: Tried but failed to load mean and std from cache. Will compute them instead."
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
            f"DownstreamDataModule: Mean and std computed over {len(all_images)} images with shape {all_images.shape}"
        )
        # Save mean and std to cache
        with open(cache_path, "w") as f:
            json.dump({"mean": mean, "std": std}, f)
            logger.debug(
                f"DownstreamDataModule: Saved mean and std to cache at {cache_path}"
            )

        return mean, std

    @override
    def get_cv_splits(self) -> Generator[Tuple[L.LightningDataModule, Tuple[float, float]], None, None]:
        """
        Generate cross-validation splits with combined training data and separate validation sets.
        Yields:
            Tuple:
                - DataModuleFolds: Lightning data module with train and validation dataloaders
                - Tuple[float, float]: Class weights (w0, w1) for handling class imbalance
        """
        for i, ((internal_train_data, internal_val_data), (btxrd_train_data, btxrd_val_data)) in enumerate(zip(self.internal_dataset.get_cv_splits(), self.btxrd_dataset.get_cv_splits())):
            logger.info(f"DownstreamDataModule: Generating K-Fold Cross Validation Split Number {i}")
            train_combined = internal_train_data + btxrd_train_data

            # for each split, compute the mean and std of the training data for normalization
            self.mean[i], self.std[i] = self._get_mean_and_std(train_combined)
            transforms_without_augmentations = Compose(
                [
                    self.transforms_before_normalization,
                    # usually usee zero mean and unit variance normalization, but for models from torchxrayvision, use the scale intensity normalization to [-1024, 1024]
                    NormalizeIntensityd(keys=["x-ray"], subtrahend=self.mean[i], divisor=self.std[i]) if not self.scale_intensity_normalization else ScaleIntensityRanged(keys=["x-ray"], a_min=0.0, a_max=255.0, b_min=-1024.0, b_max=1024.0)
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
                train_combined = np.random.choice(
                    train_combined,
                    size=self.try_with_only_n_samples,
                    replace=False,
                ).tolist()
                internal_val_data = np.random.choice(
                    internal_val_data,
                    size=self.try_with_only_n_samples,
                    replace=False,
                ).tolist()
                btxrd_val_data = np.random.choice(
                    btxrd_val_data,
                    size=self.try_with_only_n_samples,
                    replace=False,
                ).tolist()
            
            train_dataset = Dataset(data=train_combined, transform=transforms_with_augmentations)
            internal_val_dataset = Dataset(data=internal_val_data, transform=transforms_without_augmentations)
            btxrd_val_dataset = Dataset(data=btxrd_val_data, transform=transforms_without_augmentations)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            internal_val_dataloader = DataLoader(
                internal_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            btxrd_val_dataloader = DataLoader(
                btxrd_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            # calculate the class weights to be later used as pos_weight in BCEWithLogitsLoss
            labels = np.array([item["tumor"] for item in train_combined])
            w0 = len(labels) / (2 * np.sum(labels == 0))
            w1 = len(labels) / (2 * np.sum(labels == 1))
            label_weights = (float(w0), float(w1))

            yield DataModuleFolds(
                train_dataloader=train_dataloader,
                val_dataloaders=[internal_val_dataloader, btxrd_val_dataloader],
            ), label_weights

    
    def test_dataloader(self, fold:int) -> DataLoader:
        """
        Creates a test DataLoader by combining internal and BTXRD test datasets.
        Args:
            fold: Fold number used for normalization statistics (ignored if scale_intensity_normalization is True)
        Returns:
            DataLoader: Combined test dataset with appropriate transforms applied
        """
        # raise Exception(
        #     "Are you sure you want to use the test dataloader??? You should only use this in the very very end for one time evaluation. Once used, there is no going back! If you are sure, comment out this line that raises the exception."
        # )
        # return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        internal_test_dataset = self.internal_dataset.get_test_dicts()
        if self.try_with_only_n_samples is not None:
            internal_test_dataset = np.random.choice(
                internal_test_dataset,
                size=self.try_with_only_n_samples,
                replace=False,
            ).tolist()
        btxrd_test_dataset = self.btxrd_dataset.get_test_dicts()
        if self.try_with_only_n_samples is not None:
            btxrd_test_dataset = np.random.choice(
                btxrd_test_dataset,
                size=self.try_with_only_n_samples,
                replace=False,
            ).tolist()

        concat_test_dataset = internal_test_dataset + btxrd_test_dataset
        if self.scale_intensity_normalization:
            logger.info("DownstreamDataModule: Using scale intensity normalization for test dataloader. Ignoring fold number for getting mean and std.")
        concat_test_dataset = Dataset(data=concat_test_dataset, transform=Compose(
                [
                    self.transforms_before_normalization,
                    NormalizeIntensityd(keys=["x-ray"], subtrahend=self.mean[fold], divisor=self.std[fold]) if not self.scale_intensity_normalization else ScaleIntensityRanged(keys=["x-ray"], a_min=0.0, a_max=255.0, b_min=-1024.0, b_max=1024.0)
                ]
            ))
        concat_test_dataloader = DataLoader(
            concat_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return concat_test_dataloader


if __name__ == "__main__":

    stratification_over_anatomy_site = False
    image_examples = False
    augmentation_examples = False
    save_all_metadata_as_csv = True

    L.seed_everything(42)  # Set seed for reproducibility

    # Test the DataModule
    downstream_data_module = DownstreamDataModule(using_crops=False, batch_size=32, num_workers=8)

    if stratification_over_anatomy_site:
        # Calculate and plot the ratios of anatomy sites across the test and train/val splits
        # NOTE: remove transformations above to speed up the process, the images are not needed for this calculation
        records = []
        for i, (fold_data_module, _) in enumerate(downstream_data_module.get_cv_splits()):
            logger.debug(f"DownstreamDataModule: K-Fold Cross Validation Split: {i}")
            
            train_loader = fold_data_module.train_dataloader()
            val_loaders = fold_data_module.val_dataloader()
            internal_val_loader, btxrd_val_loader = val_loaders[0], val_loaders[1]

            for j, train_batch in enumerate(train_loader):
                anatomy_sites = train_batch["anatomy_site"]
                train_key = f"train_{i}"
                batch_records = list(zip(anatomy_sites, [train_key] * len(anatomy_sites)))
                records.extend(batch_records)
            logger.debug(f"DownstreamDataModule: Added anatomy sites from train for split {i}")

            for j, val_batch in enumerate(internal_val_loader):
                anatomy_sites = val_batch["anatomy_site"]
                val_key = f"val_{i}"
                batch_records = list(zip(anatomy_sites, [val_key] * len(anatomy_sites)))
                records.extend(batch_records)
            for j, val_batch in enumerate(btxrd_val_loader):
                anatomy_sites = val_batch["anatomy_site"]
                val_key = f"val_{i}"
                batch_records = list(zip(anatomy_sites, [val_key] * len(anatomy_sites)))
                records.extend(batch_records)
            logger.debug(f"DownstreamDataModule: Added anatomy sites from for split {i}")

        internal_test_dicts = downstream_data_module.internal_dataset.get_test_dicts()
        internal_test_anatomy_sites = [item["anatomy_site"] for item in internal_test_dicts]
        internal_test_records = list(zip(internal_test_anatomy_sites, ["test"] * len(internal_test_anatomy_sites)))
        records.extend(internal_test_records)
        btxrd_test_dicts = downstream_data_module.btxrd_dataset.get_test_dicts()
        btxrd_test_anatomy_sites = [item["anatomy_site"] for item in btxrd_test_dicts]
        btxrd_test_records = list(zip(btxrd_test_anatomy_sites, ["test"] * len(btxrd_test_anatomy_sites)))
        records.extend(btxrd_test_records)

        df = pd.DataFrame(records, columns=["anatomy_site", "dataset"])
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
        plt.savefig("visualizations/data/downstream/stratification_over_anatomy_site.png", dpi=300)
    
    if image_examples or augmentation_examples:
        fold_data_module, _ = next(downstream_data_module.get_cv_splits())
        train_loader = fold_data_module.train_dataloader()
        train_batch = next(iter(train_loader))

        # get the first n images from each dataset
        n = 5
        internal_examples = []
        i = 0
        while len(internal_examples) < n:
            sample_dataset = train_batch["dataset"][i]
            if sample_dataset == "INTERNAL":
                internal_examples.append({
                    "x-ray": train_batch["x-ray"][i],
                    "image_path": train_batch["image_path"][i],
                    "tumor": train_batch["tumor"][i],
                    "anatomy_site": train_batch["anatomy_site"][i],
                    "dataset": train_batch["dataset"][i],
                    "age": train_batch["age"][i],
                    "sex": train_batch["sex"][i],
                    "entity": train_batch["entity"][i],
                })
            i += 1
        logger.debug(f"DownstreamDataModule: Found {len(internal_examples)} INTERNAL examples")
        btxrd_examples = []
        i = 0
        while len(btxrd_examples) < n:
            sample_dataset = train_batch["dataset"][i]
            if sample_dataset == "BTXRD":
                btxrd_examples.append({
                    "x-ray": train_batch["x-ray"][i],
                    "image_path": train_batch["image_path"][i],
                    "tumor": train_batch["tumor"][i],
                    "anatomy_site": train_batch["anatomy_site"][i],
                    "dataset": train_batch["dataset"][i],
                    "age": train_batch["age"][i],
                    "sex": train_batch["sex"][i],
                    "entity": train_batch["entity"][i],
                })
            i += 1
        logger.debug(f"DownstreamDataModule: Found {len(btxrd_examples)} BTXRD examples")


    if image_examples:
        # Plot the images in internal/btxrd_examples
        plt.figure(figsize=(15, 5))
        for i, example in enumerate(internal_examples):
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
                f"INTERNAL\nTumor: {example['tumor']}\n"
                f"Anatomy Site: {example['anatomy_site']}\n"
                f"Age: {example['age']}\nSex: {example['sex']}\n"
                f"Entity: {example['entity']}"
            )
            ax1.set_title(title, fontsize=8)

        for i, example in enumerate(btxrd_examples):
            ax1 = plt.subplot(2, n * 2, n * 2 + i * 2 + 1)
            img = Image.open(example["image_path"])
            ax1.imshow(img, cmap="gray")
            ax1.axis("off")

            ax2 = plt.subplot(2, n * 2, n * 2 + i * 2 + 2)
            ax2.imshow(example["x-ray"][0].numpy().T, cmap="gray")
            ax2.axis("off")
            

            title = (
                f"BTXRD\nTumor: {example['tumor']}\n"
                f"Anatomy Site: {example['anatomy_site']}\n"
                f"Age: {example['age']}\nSex: {example['sex']}\n"
                f"Entity: {example['entity']}"
            )
            ax1.set_title(title, fontsize=8)

        plt.tight_layout()
        plt.savefig("visualizations/data/downstream/image_examples_with_histogram_equalization.png", dpi=300)

    if augmentation_examples:
        # NOTE: in order for this to work, make sure that the augmentations in the DownstreamDataModule are not applied to the dataset (e.g. comment them out)
        # NOTE: also, make sure that the augmentations here match the ones in the DownstreamDataModule

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
        internal_examples_augmented = []
        for internal_sample in internal_examples:
            internal_sample_augmented = []
            for i in range(4):
                internal_sample_augmented.append(augmentations(internal_sample))
            internal_examples_augmented.append(internal_sample_augmented)
        btxrd_examples_augmented = []
        for btxrd_sample in btxrd_examples:
            btxrd_sample_augmented = []
            for i in range(4):
                btxrd_sample_augmented.append(augmentations(btxrd_sample))
            btxrd_examples_augmented.append(btxrd_sample_augmented)

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
                tumor_label = sample["tumor"]

                # First column: original image
                ax = axes[row, 0] if n_samples > 1 else axes[0]
                ax.imshow(original_image[0], cmap="gray")
                ax.set_title(f"Original\nTumor: {tumor_label}")
                ax.axis("off")

                # Augmented images
                for col in range(n_augmentations):
                    augmented_image = augmented_samples[row][col]["x-ray"]
                    ax = axes[row, col + 1] if n_samples > 1 else axes[col + 1]
                    ax.imshow(augmented_image[0], cmap="gray")
                    ax.set_title(f"Aug {col + 1}\nTumor: {tumor_label}")
                    ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"visualizations/data/downstream/augmented_samples_{title_prefix}.png", dpi=300)

        plot_augmented_samples(internal_examples, internal_examples_augmented, "INTERNAL")
        plot_augmented_samples(btxrd_examples, btxrd_examples_augmented, "BTXRD")


    if save_all_metadata_as_csv:
        internal_train_val_dicts = downstream_data_module.internal_dataset.train_val_dicts
        internal_train_val_df = pd.DataFrame(internal_train_val_dicts)
        internal_train_val_df['set'] = 'train/val'
        internal_train_val_df.drop('patient_number', inplace=True, axis=1) # match the structures of the dataset dicts
        internal_test = downstream_data_module.internal_dataset.test_dicts
        internal_test_df = pd.DataFrame(internal_test)
        internal_test_df['set'] = 'test'
        btxrd_train_val_dicts = downstream_data_module.btxrd_dataset.train_val_dicts
        btxrd_train_val_df = pd.DataFrame(btxrd_train_val_dicts)
        btxrd_train_val_df['set'] = 'train/val'
        btxrd_test = downstream_data_module.btxrd_dataset.test_dicts
        btxrd_test_df = pd.DataFrame(btxrd_test)
        btxrd_test_df['set'] = 'test'
        
        total_df = pd.concat([internal_train_val_df, internal_test_df, btxrd_train_val_df, btxrd_test_df], axis=0)
        total_df.drop(['x-ray', 'image_path', 'anatomy_site_encoded', 'sex_encoded', 'age_encoded'], axis=1, inplace=True)

        total_df.to_csv('visualizations/data/downstream/metadata.csv')

