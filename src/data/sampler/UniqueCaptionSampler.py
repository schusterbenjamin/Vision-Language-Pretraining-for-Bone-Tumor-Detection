import math
from torch.utils.data import Sampler, Dataset
import numpy as np
import random

import logging
import logging.config

from tqdm import tqdm

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class NoDuplicateCaptionSampler(Sampler):
    """
    A PyTorch sampler that tries to prevent duplicated captions within each batch.

    Args
    ----------
        dataset: The dataset to sample from
        batch_size (int): Number of samples per batch
        caption_ids (list): List of caption IDs corresponding to each dataset sample.
            Can contain any integers (e.g., [454, 13, 92, 454]) which will be
            internally mapped to consecutive indices (e.g., [0, 1, 2, 0])
        probabilistic_mode (str, optional): Sampling strategy for caption selection.
            - "full": Select captions probabilistically based on sample count
            - "semi": Select captions deterministically (highest sample count first)
            Defaults to "full"
        deterministic (bool, optional): If True, caches batches for reproducible
            iteration. Defaults to False

    Raises
    ----------
        AssertionError: If caption_ids length doesn't match dataset length or
            if probabilistic_mode is not "full" or "semi"

    Notes:
        Behavior:
            - Creates batches with unique captions when possible
            - When fewer unique captions remain than batch_size, captions with duplicates will be produced
            - In final batches, may include duplicate captions if necessary to maintain
            batch_size
            - Removes samples once used to prevent reuse within the same epoch
            - In deterministic mode, caches all batches for consistent iteration
    """
    def __init__(self, dataset, batch_size, caption_ids, probabilistic_mode="full", deterministic=False):
        
        assert len(caption_ids) == len(dataset), f"caption_ids must have the same length as the dataset. Length of caption_ids: {len(caption_ids)}, Length of dataset: {len(dataset)}"
        assert probabilistic_mode in ["full", "semi"], f"probabilistic_mode must be either 'full' or 'semi'. Got: {probabilistic_mode}"

        self.dataset = dataset
        self.batch_size = batch_size

        # caption_ids can have any numbers e.g. [454, 13, 92, 454], I want to translate them to [0, 1, 2, 0]
        unique_caption_ids = list(set(caption_ids))
        caption_id_map = {original_id: idx for idx, original_id in enumerate(unique_caption_ids)}
        self.caption_ids = [caption_id_map[original_id] for original_id in caption_ids]

        self.number_of_unique_captions = len(unique_caption_ids)

        self.probabilistic_mode = probabilistic_mode

        self.deterministic = deterministic
        if deterministic:
            self.cached_batches = []

        logger.info(
            f"NoDuplicateCaptionSampler: Successfully initialized with {self.number_of_unique_captions} unique captions."
        )

    def __iter__(self):
        if self.deterministic:
            # if deterministic mode is on, then check if we have already cached the batches
            if len(self.cached_batches) == len(self): # if we have already cached all the batches, just return the cached batches
                logger.debug("NoDuplicateCaptionSampler: Returning cached batches.")
                for batch in self.cached_batches:
                    yield batch
                return
            else:
                self.cached_batches = [] # reset cached batches if we haven't cached all the batches yet, this is important since sanity checking inflates the cache before it is created fully for the first time
            
        
        available_samples = {}
        for caption_id in range(self.number_of_unique_captions):
            # get the indeces of elements in self.caption_ids that match the current caption_id
            indices = [i for i, cid in enumerate(self.caption_ids) if cid == caption_id]
            if len(indices) > 0: # only include caption_ids that have samples
                available_samples[caption_id] = indices

        # Create batches until we don't have enough unique captions
        while len(available_samples) >= self.batch_size:
            caption_ids = list(available_samples.keys())
            
            if self.probabilistic_mode == "full":
                # Calculate weights for each caption ID based on the number of samples
                weights = [len(available_samples[caption_id]) for caption_id in caption_ids]
                # Normalize weights to probabilities
                total_weight = sum(weights)
                probabilities = [weight / total_weight for weight in weights]
                
                # Select batch_size caption IDs with probabilities proportional to their sample count
                selected_caption_ids = np.random.choice(
                    caption_ids,
                    size=self.batch_size,
                    replace=False,  # Without replacement to ensure unique captions
                    p=probabilities
                )
            else:
                # select the caption ids with the highest number of samples
                caption_ids.sort(key=lambda cid: len(available_samples[cid]), reverse=True)
                selected_caption_ids = caption_ids[: self.batch_size]

            # For each selected caption ID, select one sample randomly
            batch = []
            for caption_id in selected_caption_ids:
                sample_index = random.choice(available_samples[caption_id])
                batch.append(sample_index)

                # Remove the selected sample from available_samples
                available_samples[caption_id].remove(sample_index)
                if not available_samples[
                    caption_id
                ]:  # If no more samples for this caption, remove the caption
                    del available_samples[caption_id]

            if self.deterministic:
                self.cached_batches.append(batch)
            yield batch

        # Handle remaining samples (less than batch_size unique captions left)
        while available_samples:
            batch = []

            # Add remaining unique captions to the batch
            for caption_id in list(available_samples.keys()):
                sample_index = random.choice(available_samples[caption_id])
                batch.append(sample_index)

                # Remove the selected sample
                available_samples[caption_id].remove(sample_index)
                if not available_samples[caption_id]:
                    del available_samples[caption_id]

            if len(batch) < self.batch_size and available_samples:
                logger.debug("NoDuplicateCaptionSampler: Forced to create a batch with duplicated captions :/")
            
            # Fill the batch with duplicate captions if necessary
            while len(batch) < self.batch_size and available_samples:
                # Find the caption with the highest number of occurrences
                caption_id = max(
                    available_samples.keys(),
                    key=lambda cid: len(available_samples[cid]),
                )

                # Add a sample from this caption
                sample_index = random.choice(available_samples[caption_id])
                batch.append(sample_index)

                # Remove the selected sample
                available_samples[caption_id].remove(sample_index)
                if not available_samples[caption_id]:
                    del available_samples[caption_id]

            # Yield the final batch
            if batch:
                if self.deterministic:
                    self.cached_batches.append(batch)
                yield batch

    def __len__(self):
        return math.ceil(len(self.caption_ids) / self.batch_size)


# NOTE: Just for testing
class CaptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


if __name__ == "__main__":
    dataset = [
        {"caption": "A cat on a roof"},
        {"caption": "A cat on a roof"},
        {"caption": "Hello there"},
        {"caption": "Hello there"},
        {"caption": "A dog in the park"},
        {"caption": "A dog in the park"},
        {"caption": "Okay"},
        {"caption": "Okay"},
        {"caption": "Okay"},
        {"caption": "Okay"},
        {"caption": "Okay"},
        {"caption": "Okay"},
        {"caption": "Okay"},
        {"caption": "Okay"}
    ]
    dataset = CaptionDataset(dataset)
    caption_ids = [6, 6, 1, 1, 252, 252, 3, 3, 3, 3, 3, 3, 3, 3]

    sampler = NoDuplicateCaptionSampler(dataset, batch_size=2, caption_ids=caption_ids, probabilistic_mode="semi", deterministic=True)

    print("sanity checking:")
    cap = 2
    for i, batch in enumerate(sampler):
        print(batch, [dataset[i]["caption"] for i in batch])
        if i >= cap:
            break

    print("First epoch:")
    for batch in sampler:
        print(batch, [dataset[i]["caption"] for i in batch])

    print("Second epoch (should be the same as the first one):")
    for batch in sampler:
        print(batch, [dataset[i]["caption"] for i in batch])

    print("Third epoch (should be the same as the first one):")
    for batch in sampler:
        print(batch, [dataset[i]["caption"] for i in batch])
