import os
import sys
from itertools import chain
import logging
import logging.config
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import timm

from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel
import lightning as L

import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.data.DownstreamDataModule import DownstreamDataModule


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")


class ImageEncoder(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model, pretrained=False, num_classes=0, global_pool="avg", **kwargs
        )

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, text_encoder_model):
        super().__init__()

        if text_encoder_model == "distilbert":
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif text_encoder_model == "tinybert":
            self.model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", torch_dtype="auto")
        else:
            raise ValueError(
                f"VisionLanguageModule: Text encoder model {text_encoder_model} is not supported. Supported models are: distilbert, tinybert."
            )

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        # ensure that the model is trainable
        self.model.train()

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class VisionLanguageModule(L.LightningModule):
    def __init__(
        self,
        image_model,
        text_encoder_model,
        optimizer: torch.optim.Optimizer,
        deduplicate: bool,
        masked_loss: bool,
        image_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        embedding_dim: int = 256,
        label_weights: tuple = (
            1.0,
            1.0,
        ),  # this is just to keep the interface consistent with the other modules, this is not used in the VLP module
        scheduler: torch.optim.lr_scheduler = None,
        downstream_datamodule: DownstreamDataModule=None,
        text_encoder_lr: float = None,
        image_encoder_lr: float = None,
        projections_lr: float = None,
        image_encoder_droupout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if deduplicate:
            if masked_loss:
                logger.warning(
                    "Deduplication and masked loss are mutually exclusive. Deduplication will be used."
                )
            masked_loss = False  # deduplication is preferred over masked loss
        
        self.save_hyperparameters(
            logger=False
        )  # the hyperparameters are already logged by the train script including also data, trainer, etc.

        self.image_encoder = ImageEncoder(image_model, drop_rate=image_encoder_droupout)
        self.text_encoder = TextEncoder(text_encoder_model)

        # initialization adapted from CLIP, see ATTRIBUTION.md for their License
        self.image_projection = nn.Parameter(
            torch.empty(image_embedding_dim, embedding_dim)
        )
        nn.init.normal_(self.image_projection, std=image_embedding_dim**-0.5)
        self.text_projection = nn.Parameter(
            torch.empty(text_embedding_dim, embedding_dim)
        )
        nn.init.normal_(self.text_projection, std=text_embedding_dim**-0.5)

        self.logit_scale = nn.Parameter(torch.tensor([np.log(1/0.07)]))

        self.deduplicated_loss_function = torch.nn.BCEWithLogitsLoss()

        self.k_for_precision_at_k = [3, 5, 10, 15]
        self.k_for_image_text_retreival = [3, 5, 10, 15]

        self.val_combined_loss = MeanMetric()

        self.downstream_datamodule = downstream_datamodule
        if self.downstream_datamodule is not None:
            dm, _ = next(self.downstream_datamodule.get_cv_splits()) # get the first split of k-fold cv
            self.downstream_train_dataloader = dm.train_dataloader()
            self.downstream_val_dataloaders = dm.val_dataloader()

        logger.info(
            f"VisionLanguageModule: Successfully initialized with the following parameters: {self.hparams}"
        )

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for the vision-language module.
        This method sets up the optimizer with parameter groups configured by 
        _configure_optimizer_parameters(), logs detailed information about each 
        parameter group including names, learning rates, and parameter counts,
        and optionally configures a learning rate scheduler.
        Returns:
            dict: Dictionary containing optimizer configuration. If scheduler is provided,
                  returns dict with 'optimizer' and 'lr_scheduler' keys. The lr_scheduler
                  dict contains the scheduler instance, interval set to 'epoch', and 
                  frequency set to 1. If no scheduler is provided, returns dict with 
                  only 'optimizer' key.
        Side Effects:
            - Logs parameter group information including names, learning rates, and counts
            - Logs total number of optimized parameters
            - Sets self.hparams["num_optimized_params"] with the total parameter count
        """

        param_groups = self._configure_optimizer_parameters()
        optimizer = self.hparams.optimizer(params=param_groups)

        logger.info(
            f"VisionLanguageModule: Optimizer configured with the following parameter groups:"
        )
        for group in optimizer.param_groups:
            name = group.get("name", "unnamed")
            lr = group.get("lr", "default")
            n_params = sum(p.numel() for p in group["params"])
            logger.info(f"Parameter group '{name}': {n_params} params, lr={lr}")
        logger.info(f"Default learning rate: {optimizer.defaults['lr']}")

        # log the number of parameters that are optimized using the optimizer
        num_optimized_params = sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        logger.info(
            f"VisionLanguageModule: Number of parameters optimized by the optimizer: {num_optimized_params}"
        )
        self.hparams["num_optimized_params"] = num_optimized_params
        

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def _configure_optimizer_parameters(self):
        """
        Configure optimizer parameter groups with different learning rates for different model components.
        This method organizes model parameters into distinct groups to allow for component-specific
        learning rates during training. It separates parameters from the image encoder, text encoder,
        projection layers, and logit scale, while handling any remaining unassigned parameters.
        Returns:
            list: A list of parameter group dictionaries, where each dictionary contains:
                - 'params': List of parameters for the group
                - 'name': String identifier for the parameter group
                - Additional optimizer-specific parameters (e.g., learning rate) if specified
        Parameter Groups:
            - remaining_params: Any parameters not explicitly assigned to other groups
            - projection_and_logitscale: Image/text projection layers and logit scale parameter
            - image_encoder: All parameters from the image encoder model
            - text_encoder: All parameters from the text encoder model
        Warnings:
            Logs a warning if there are unassigned parameters, which may indicate
            architecture or configuration mismatches.
        Note:
            Parameter groups are only added if they contain parameters and have valid
            configurations as determined by _get_param_group().
        """
        image_encoder_params = list(self.image_encoder.parameters())
        text_encoder_params = list(self.text_encoder.parameters())
        projection_and_logitscale_param_group = [self.image_projection, self.text_projection, self.logit_scale]
        assigned_params = set(image_encoder_params + text_encoder_params + projection_and_logitscale_param_group)
        remaining_params = [p for p in self.parameters() if p not in assigned_params]

        # set specific learning rates for different parts of the model if specified otherwise use default
        param_groups = []
        group = {"params": remaining_params, "name": "remaining_params"}
        # print a warning if there are any remaining parameters that are not assigned to any group
        if len(remaining_params) > 0:
            logger.warning(
                f"VisionLanguageModule: There are {len(remaining_params)} parameters that are not assigned to any group. This might be due to a mismatch in the model architecture or the optimizer configuration."
            )
        param_groups.append(group)

        projection_and_logitscale_param_group = self._get_param_group(
            projection_and_logitscale_param_group, "projection_and_logitscale", self.hparams.projections_lr
        )
        if projection_and_logitscale_param_group is not None:
            param_groups.append(projection_and_logitscale_param_group)

        image_encoder_param_group = self._get_param_group(
            image_encoder_params, "image_encoder", self.hparams.image_encoder_lr
        )
        if image_encoder_param_group is not None:
            param_groups.append(image_encoder_param_group)
        
        text_encoder_param_group = self._get_param_group(
            text_encoder_params, "text_encoder", self.hparams.text_encoder_lr
        )
        if text_encoder_param_group is not None:
            param_groups.append(text_encoder_param_group)

        return param_groups

    def _get_param_group(self, params, name: str, lr: float | None):
        """
        Create a parameter group for optimizer configuration with optional learning rate scaling.

        This method creates a parameter group dictionary that can be used with PyTorch optimizers,
        with support for learning rate scaling and parameter freezing.

        Args:
            params: List of model parameters to include in the group
            name (str): Name identifier for the parameter group (used for logging)
            lr (float | None): Learning rate for this parameter group. If None, uses default
                              optimizer learning rate. If 0, parameters are frozen and group
                              is excluded from optimizer. Must be non-negative.

        Returns:
            dict | None: Parameter group dictionary with 'params', 'name', and optionally 'lr' keys.
                        Returns None if lr=0 (frozen parameters should not be added to optimizer).

        Raises:
            ValueError: If lr is negative.

        Note:
            When lr=0, the parameters in the group are frozen (requires_grad=False) and the
            method returns None to indicate the group should not be added to the optimizer.
        """
        group = {"params": params, "name": name}
        if lr is not None:
            group["lr"] = lr
            if lr == 0:
                logger.info(
                    f"VisionLanguageModule: {name} scale learning rate is set to 0. This means that these parameters will not be updated during training."
                )
                # freeze the parameters
                for param in params:
                    param.requires_grad = False

                return None # if lr is 0, we don't want to add this group to the optimizer
            elif lr < 0:
                logger.error(
                    f"VisionLanguageModule: {name} scale learning rate is set to a negative value."
                )
                raise ValueError(
                    f"VisionLanguageModule: {name} scale learning rate must be a non-negative value."
                )
            else:
                logger.info(
                    f"VisionLanguageModule: {name} scale learning rate is set to {lr}."
                )
        else:
            logger.info(
                f"VisionLanguageModule: {name} scale learning rate is not set. Using default learning rate."
            )
        return group
    
    def evaluate_downstream_precision_at_k(self, mode='entire') -> Tuple[dict, dict]:
        """
        Evaluate precision@k on downstream task using image embeddings.
        This method computes precision@k metrics by collecting image embeddings and their
        corresponding labels from the downstream dataset, then calculating how well the
        model can retrieve relevant images based on similarity in the embedding space.
        Args:
            mode (str, optional): Evaluation mode specifying which data to use.
                - 'entire': Use both training and validation data for evaluation
                - 'validation': Use only validation data for evaluation
                Defaults to 'entire'.
        Returns:
            Tuple[dict, dict]: Precision@k results containing metrics for different k values.
                The exact structure depends on the implementation of precision_at_k_on_image_embeddings.
        Raises:
            ValueError: If mode is not 'entire' or 'validation'.
        Note:
            - Sets model to evaluation mode during computation and restores training mode afterward
            - Uses gradient computation disabled for efficiency
            - Processes batches with 'tumor' labels and 'x-ray' images
            - Accumulates all embeddings in memory before computing precision@k
        """
        # collect image_embeddings and labels for precision at k computation
        image_embeddings_and_labels = {
            'images': torch.empty(0, device=self.device),
            'labels': torch.empty(0, device=self.device, dtype=torch.int64)
        }

        self.eval()
        with torch.no_grad():
            # use all training and validation data for zero-shot evaluation
            if mode == 'entire':
                all_batches = chain(self.downstream_train_dataloader, *self.downstream_val_dataloaders)
            elif mode == 'validation':
                all_batches = chain(*self.downstream_val_dataloaders)
            else:
                raise ValueError(f"Invalid mode: {mode}. Supported modes are: 'entire', 'validation'.")
            
            for batch in tqdm(all_batches, desc=f"Label precision@k on {mode} downstream dataset"):
                batch['tumor'] = batch['tumor'].to(device=self.device, dtype=torch.int64)
                batch['x-ray']= batch['x-ray'].to(device=self.device)

                labels = batch['tumor']

                image_features = self.image_encoder(batch["x-ray"])
                image_embeddings = image_features @ self.image_projection

                # collect image embeddings and labels for precision at k computation
                image_embeddings_and_labels['images'] = torch.cat(
                    (image_embeddings_and_labels['images'], image_embeddings), dim=0
                )
                image_embeddings_and_labels['labels'] = torch.cat(
                    (image_embeddings_and_labels['labels'], labels), dim=0
                )
        self.train()

        downstream_precision_at_k = self.precision_at_k_on_image_embeddings(
            image_embeddings_and_labels['images'],
            image_embeddings_and_labels['labels'],
            ks=self.k_for_precision_at_k
        )

        return downstream_precision_at_k

    
    def precision_at_k_on_image_embeddings(self, image_embeddings, labels, ks:list) -> dict:
        """
        Compute precision@k for image embeddings based on cosine similarity.
        This method calculates how often the k most similar images (by cosine similarity)
        to each query image have the same label as the query image.
        Args:
            image_embeddings (torch.Tensor): Normalized image embeddings of shape [batch_size, embedding_dim].
            labels (torch.Tensor): Ground truth labels for each image of shape [batch_size].
            ks (list): List of k values to compute precision@k for.
        Returns:
            dict: Dictionary mapping each k value to its corresponding precision@k score (float).
                    The precision@k is averaged across all images in the batch.
        Raises:
            AssertionError: If any k+1 is greater than the batch size (not enough samples for comparison).
        Note:
            - Image embeddings are normalized to unit vectors for cosine similarity computation
            - Precision@k = (number of correct predictions in top-k) / k
        """
        assert all(k + 1 <= image_embeddings.shape[0] for k in ks), "k+1 must be less than or equal to the batch size"

        # compute the cosine similarity between all image embedding pairs
        image_embeddings = torch.nn.functional.normalize(image_embeddings) # they should already be normalized, but just to be sure
        similarity_matrix = image_embeddings @ image_embeddings.T  # [batch_size, batch_size]
        
        precision_at_k_result = {}
        for k in ks:
            # get the top k indices for each image embedding
            top_k_indices = similarity_matrix.topk(k=k+1, dim=1).indices  # [batch_size, k]
            # remove the first index, which is the image itself (self-similarity)
            top_k_indices = top_k_indices[:, 1:]  # [batch_size, k
            # check if the labels of the top k indices match the labels of the current image embedding
            correct_predictions = (labels.unsqueeze(1) == labels[top_k_indices]).sum(dim=1)  # [batch_size]
            # compute the precision at k
            precision_at_k = correct_predictions.float() / k  # [batch_size]
            precision_at_k_result[k] = precision_at_k.mean().item()
        
        return precision_at_k_result
    
    def recall_at_k_on_image_text_retreival(self, image_embeddings, text_embeddings, ks:list) -> dict:
        """
        Calculate recall@k metrics for image-text retrieval task.
        Given image embeddings, this function retrieves the top k most similar text embeddings
        and measures how often the correct (aligned) text embedding appears in the top k results.
        Args:
            image_embeddings (torch.Tensor): Normalized image embeddings of shape [batch_size, embedding_dim]
            text_embeddings (torch.Tensor): Normalized text embeddings of shape [batch_size, embedding_dim]
            ks (list): List of k values to compute recall@k for (e.g., [1, 3, 5, 10])
        Returns:
            dict: Dictionary mapping each k value to its corresponding recall@k score.
                Recall@k is the fraction of images for which the correct text embedding
                appears in the top k retrieved text embeddings.
        Note:
            - Image and text embeddings are assumed to be aligned (i-th image corresponds to i-th text)
        """
        # Explanation of image-text recall@3
        # given an image, retreive the top 3 most similar text embeddings
        # think 1 if the correct text embedding is in the top 3, else 0
        # average over all images
        # we can assume, that the list of image embeddings and text embeddings are aligned, meaning that the i-th image embedding corresponds to the i-th text embedding

        image_embeddings = torch.nn.functional.normalize(image_embeddings)
        text_embeddings = torch.nn.functional.normalize(text_embeddings)
        similarity_matrix = image_embeddings @ text_embeddings.T  # [batch_size, batch_size], perfect matrix would be 1 on diagonal and 0 elsewhere

        recall_at_k_result = {}
        for k in ks:
            top_k_indices = similarity_matrix.topk(k=k, dim=1).indices  # [batch_size, k]
            targets = torch.arange(image_embeddings.shape[0], device=top_k_indices.device)
            correct_embedding_in_top_k = (top_k_indices == targets.unsqueeze(1)).any(dim=1)

            correct_embeddings_in_top_k_sum = correct_embedding_in_top_k.sum().item()
            recall_at_k = correct_embeddings_in_top_k_sum / image_embeddings.shape[0]

            recall_at_k_result[k] = recall_at_k
        
        return recall_at_k_result

    def forward(self, batch):
        # foward pass adapted from CLIP, see ATTRIBUTION.md for their License
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["x-ray"])
        text_features = self.text_encoder(**batch["caption_tokenized"])

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = image_features @ self.image_projection
        text_embeddings = text_features @ self.text_projection

        # Normalize the embeddings
        image_embeddings = torch.nn.functional.normalize(image_embeddings)
        text_embeddings = torch.nn.functional.normalize(text_embeddings)

        # Calculating the logits
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100) # clamp according to this github issue: https://github.com/openai/CLIP/issues/46#issuecomment-782558799

        logits = (image_embeddings @ text_embeddings.T) * logit_scale

        return logits, image_embeddings, text_embeddings
    

    def _compute_non_square_loss(self, logits, captions):
        """
        Compute loss for non-square similarity matrices where there are duplicate captions.
        This method handles cases where the same caption appears multiple times in a batch,
        creating a non-square loss computation by deduplicating captions and creating
        appropriate labels for multi-class classification.
        Args:
            logits (torch.Tensor): Similarity logits matrix of shape (batch_size, num_captions)
            captions (list or array): List of caption strings, may contain duplicates
        Returns:
            torch.Tensor: Computed loss value using the deduplicated loss function
        Process:
            1. Identifies unique captions and their occurrence indices
            2. Reduces logits matrix by selecting only unique caption columns
            3. Creates binary labels matrix where each row corresponds to images
               and columns to unique captions, with 1.0 for matching pairs
            4. Computes loss using the deduplicated loss function
        """
        # get duplicate captions
        _, caption_ids = np.unique(captions, return_inverse=True) # get the uniqueness by captions and not the embeddings, since they might actually differ due to dropout during training
        # turn into torch tensor
        caption_ids = torch.tensor(caption_ids, dtype=torch.int64)

        # there are multiple "classes" now to which we should maximize, we get them by getting the indices of the unique captions
        unique_vals = torch.unique(caption_ids)
        class_indices = [(caption_ids == val).nonzero(as_tuple=True)[0].tolist() for val in unique_vals]

        # given the unique caption ids, remove duplicate columns
        unique_ids = torch.unique(caption_ids, return_inverse=False, return_counts=False, sorted=False)
        # For each unique id, get the FIRST index where it occurs
        selected_indices = torch.stack([torch.where(caption_ids == uid)[0][0] for uid in unique_ids])
        # Select logits
        selected_logits = logits[:, selected_indices]

        
        labels = torch.zeros_like(selected_logits)
        for class_id, indices in enumerate(class_indices):
            labels[indices, class_id] = 1.0

        loss = self.deduplicated_loss_function(selected_logits, labels)
        return loss
    
    def _get_mask(self, captions):
        """
        Create a mask to ignore similarities between duplicate captions.
        This method generates a mask matrix that can be applied to the similarity logits
        to zero out entries corresponding to duplicate captions, preventing them from
        contributing to the loss computation.
        Args:
            captions (list or array): List of caption strings, may contain duplicates
        Returns:
            torch.Tensor: Mask matrix of shape (N, N) where N is the number of captions.
                          Entries are 0.0 for duplicate caption pairs and 1.0 otherwise.
        """

        # Step 1: Convert strings to unique indices
        unique_captions = {caption: idx for idx, caption in enumerate(set(captions))}
        caption_ids = torch.tensor([unique_captions[c] for c in captions])

        # Step 2: Create comparison matrix
        eq = caption_ids.unsqueeze(0) == caption_ids.unsqueeze(1)  # shape: (N, N)

        # Step 3: Create mask
        mask = torch.ones_like(eq, dtype=torch.float, device=self.device)
        mask[eq & ~torch.eye(len(captions), dtype=torch.bool)] = 0.0

        return mask

    def _compute_loss(self, logits, deduplicate: bool = True, masked: bool = False, captions: list = None):
        labels = torch.arange(len(logits), device=logits.device)

        if deduplicate:
            raise DeprecationWarning(
                "Deduplication loss was made obsolete by generating diverse captions and the custom batch sampler"
            )
            assert captions is not None, "captions must be provided if deduplicate is True"
            return self._compute_non_square_loss(logits, captions)
        if masked:
            raise DeprecationWarning(
                "Masked loss was made obsolete by generating diverse captions and the custom batch sampler"
            )
            assert captions is not None, "captions must be provided if masked is True"
            mask = self._get_mask(captions)
            logits = logits * mask

        # Compute the loss using cross-entropy
        image_loss = F.cross_entropy(logits, labels, reduction="mean")
        text_loss = F.cross_entropy(logits.T, labels, reduction="mean")
        loss = (image_loss + text_loss) / 2

        return loss, image_loss, text_loss

    def _cache_embeddings_and_labels(self, image_embeddings, text_embeddings, labels, mode):
        """
        Cache image embeddings, text embeddings, and labels for a specific dataset split.

        This is done to compute retrieval metrics over the entire dataset at the end of an epoch.
        
        This method accumulates embeddings and labels across multiple batches by concatenating
        them to existing cached tensors. If cache entries don't exist, they are initialized
        as empty tensors with appropriate device and dtype.
        
        Args:
            image_embeddings (torch.Tensor): Image embeddings tensor to cache
            text_embeddings (torch.Tensor): Text embeddings tensor to cache  
            labels (torch.Tensor): Labels tensor to cache
            mode (str): Dataset split mode, must be one of ["train", "val"]
            
        Raises:
            AssertionError: If mode is not one of the valid options
            
        Note:
            The cached tensors are stored in instance attributes based on the mode:
            - "train": self.train_image_embeddings_and_labels_cached
            - "val": self.val_image_embeddings_and_labels_cached 
        """
        assert mode in ["train", "val"], f"Invalid mode: {mode}"

        if mode == "train":
            cache = self.train_image_embeddings_and_labels_cached
        elif mode == "val":
            cache = self.val_image_embeddings_and_labels_cached

        if "image_embedding" not in cache:
            cache["image_embedding"] = torch.empty(0, device=self.device, dtype=image_embeddings.dtype)
        cache["image_embedding"] = torch.cat(
            (cache["image_embedding"], image_embeddings), dim=0
        )
        if "text_embedding" not in cache:
            cache["text_embedding"] = torch.empty(0, device=self.device, dtype=text_embeddings.dtype)
        cache["text_embedding"] = torch.cat((cache["text_embedding"], text_embeddings), dim=0)
        if "label" not in cache:
            cache["label"] = torch.empty(0, device=self.device, dtype=labels.dtype)
        cache["label"] = torch.cat((cache["label"], labels), dim=0)

    def _get_cached_embeddings_and_labels(self, mode):
        """
        Retrieve cached image embeddings, text embeddings, and labels for the specified mode.
        
        Args:
            mode (str): The dataset mode to retrieve cached data for. Must be one of 
                       "train", "val".
        
        Returns:
            tuple: A tuple containing:
                - image_embedding: Cached image embeddings for the specified mode
                - text_embedding: Cached text embeddings for the specified mode  
                - label: Cached labels for the specified mode
        
        Raises:
            AssertionError: If mode is not one of "train", "val".
            ValueError: If no cached embeddings and labels exist for the specified mode,
                       or if required keys "image_embedding" or "label" are missing from cache.
        """
        assert mode in ["train", "val"], f"Invalid mode: {mode}"

        if mode == "train":
            cache = self.train_image_embeddings_and_labels_cached
        elif mode == "val":
            cache = self.val_image_embeddings_and_labels_cached

        if "image_embedding" not in cache or "label" not in cache:
            raise ValueError(f"No cached embeddings and labels for mode: {mode}")

        return cache["image_embedding"], cache["text_embedding"], cache["label"]


    def on_train_epoch_start(self):
        self.train_image_embeddings_and_labels_cached = {}

    def training_step(self, batch):
        logits, image_embeddings, text_embeddings = self(batch)
        self._cache_embeddings_and_labels(image_embeddings, text_embeddings, batch["label"], mode="train") # cache image embeddings and labels for retrieval over entire training dataset

        loss, image_loss, text_loss = self._compute_loss(logits, self.hparams['deduplicate'], self.hparams['masked_loss'], batch['caption'])

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch["x-ray"].shape[0])
        # self.log("train/image_loss", image_loss, on_step=True, on_epoch=True, batch_size=batch["x-ray"].shape[0])
        # self.log("train/text_loss", text_loss, on_step=True, on_epoch=True, batch_size=batch["x-ray"].shape[0])
        self.log("logit_scale", self.logit_scale.exp(), on_step=True, on_epoch=True, batch_size=batch["x-ray"].shape[0])
        
        return loss
    
    def on_train_epoch_end(self):
        image_embeddings, text_embeddings, labels = self._get_cached_embeddings_and_labels(mode="train")

        label_precision_at_k = self.precision_at_k_on_image_embeddings(image_embeddings, labels, ks=self.k_for_precision_at_k)
        for k, v in label_precision_at_k.items():
            self.log(f"train/label_precision_at_{k}", v, on_step=False, on_epoch=True, batch_size=image_embeddings.shape[0])

        image_text_recall_at_k = self.recall_at_k_on_image_text_retreival(image_embeddings, text_embeddings, ks=self.k_for_image_text_retreival)
        for k, v in image_text_recall_at_k.items():
            self.log(f"train/image_text_recall_at_{k}", v, on_step=False, on_epoch=True, batch_size=image_embeddings.shape[0])
        
    
    def on_validation_epoch_start(self):
        self.val_combined_loss.reset()
        self.val_image_embeddings_and_labels_cached = {}


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, image_embeddings, text_embeddings = self(batch)
        self._cache_embeddings_and_labels(image_embeddings, text_embeddings, batch["label"], mode="val") # cache image embeddings and labels for retrieval over entire validation dataset

        loss, image_loss, text_loss = self._compute_loss(logits, self.hparams['deduplicate'], self.hparams['masked_loss'], batch['caption'])

        # We assume that the first dataloader is for the LERA dataset and the second dataloader for the MURA dataset
        if dataloader_idx == 0:
            log_path_str = "val/lera"
        elif dataloader_idx == 1:
            log_path_str = "val/mura"
        else:
            raise ValueError(
                f"VisionLanguageModule: Validation dataloader index {dataloader_idx} is not supported. Supported indices are: 0, 1. We are assuming that the first dataloader is for the LERA dataset and the second dataloader for the MURA dataset"
            )

        self.log(f"{log_path_str}/loss", loss, on_step=False, on_epoch=True, batch_size=batch["x-ray"].shape[0], add_dataloader_idx=False)
        self.val_combined_loss.update(loss, batch["x-ray"].shape[0])

        return loss
    
    def on_validation_epoch_end(self):
        self.log("val/combined/loss", self.val_combined_loss.compute(), prog_bar=True)

        image_embeddings, text_embeddings, labels = self._get_cached_embeddings_and_labels(mode="val")
        label_precision_at_k = self.precision_at_k_on_image_embeddings(image_embeddings, labels, ks=self.k_for_precision_at_k)
        for k, v in label_precision_at_k.items():
            self.log(f"val/combined/label_precision_at_{k}", v, batch_size=image_embeddings.shape[0])
        image_text_recall_at_k = self.recall_at_k_on_image_text_retreival(image_embeddings, text_embeddings, ks=self.k_for_image_text_retreival)
        for k, v in image_text_recall_at_k.items():
            self.log(f"val/combined/image_text_recall_at_{k}", v, batch_size=image_embeddings.shape[0])

        # Evaluate downstream zero-shot metrics on downstream validation set

        # dont do it during sanity check, since it is not needed and takes a lot of time
        if self.trainer.sanity_checking:
            logger.info("VisionLanguageModule: Skipping downstream zero-shot evaluation during sanity check.")
            return
        downstream_val_precision_at_k = self.evaluate_downstream_precision_at_k(mode='validation')
        if downstream_val_precision_at_k:
            for k, v in downstream_val_precision_at_k.items():
                self.log(f"downstream_validation/label_precision_at_{k}", v, on_step=False, on_epoch=True)
        


if __name__ == "__main__":
    # Dummy input
    batch_size = 30
    x = torch.randn(
        batch_size, 3, 224, 224
    )  # Batch size of 1, 1 channels, 224x224 image
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_captions = tokenizer(
        ["This is a test caption."] * batch_size,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=40,
    )
    batch = {
        "x-ray": x,
        "caption_tokenized": tokenized_captions,
        'label': torch.randint(0, 2, (batch_size,)),
        'caption': ["This is a test caption."] * batch_size,
    }

    model = VisionLanguageModule(image_model="resnet34", optimizer=torch.optim.Adam, text_encoder_model="distilbert", deduplicate=False, masked_loss=False, text_embedding_dim=768, image_embedding_dim=512, embedding_dim=32)
    model.on_train_epoch_start()
    model.training_step(batch)
    model.on_train_epoch_end()

    model.on_validation_epoch_start()
    model.validation_step(batch, batch_idx=0)

    print("VisionLanguageModule test run successful.")
