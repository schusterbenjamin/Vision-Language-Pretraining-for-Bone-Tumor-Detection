import os
import sys
from typing import Tuple
import lightning as L
import torch
import timm
import torchxrayvision as xrv

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)

import logging.config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.coral_loss.coral import coral

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("project")

supported_models = [
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "resnet50",
    "resnet34",
    "nest_small",
    "resnet50-res512-all"
]


class OnlyImagingModule(L.LightningModule):
    def __init__(
        self,
        model: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        label_weights: Tuple[float] = (1.0, 1.0), # this will be set by the train script and should NOT be configured in the hydra config!
        coral_lambda: float = 0.0,  # lambda for the coral loss
        pretrained_vlp_module: str = None, # if a pretrained vision language module is provided, we use its vision encoder as pretrained image model
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False) # the hyperparameters are already logged by the train script including also data, trainer, etc.

        if model not in supported_models:
            raise ValueError(
                f"OnlyImagingModule: Model {model} is not supported. Supported models are: {supported_models}"
            )

        # catch special case for second baseline: resnet50 pretrained on chest x-rays from torchxrayvision
        if model == "resnet50-res512-all":
            class PretrainedResnet(torch.nn.Module):
                def __init__(self):
                    super(PretrainedResnet, self).__init__()
                    self.vision_encoder = xrv.models.ResNet(weights="resnet50-res512-all", cache_dir='./.cache/torchxrayvision/')
                    self.classififer = torch.nn.Linear(in_features=2048, out_features=1)
                def forward(self, x):
                    return self.classififer(self.vision_encoder.features(x))
                def forward_features(self, x):
                    return self.vision_encoder.features(x)
                def forward_head(self, x):
                    return self.classififer(x)
                
            self.network = PretrainedResnet()
            logger.info("OnlyImagingModule: Using resnet50-res512-all from torchxrayvision as image model.")
        # if no pretrained vision language module is provided, train from scratch
        elif pretrained_vlp_module is None:
            # We do binary classification, so we set num_classes=1
            self.network = timm.create_model(model, num_classes=1)
        # if a pretrained vision language module is provided, we use its vision encoder as pretrained image model
        else:
            checkpoint = torch.load(pretrained_vlp_module, map_location="cpu", weights_only=False)
            # get all key value pairst that start with "image_encoder.model." and remove the rest
            state_dict = {k.replace("image_encoder.model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("image_encoder.model.")}
            self.network = timm.create_model(model, num_classes=1, pretrained=False)
            missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
             # get how many params have been loaded
            total_params = 0
            for v in state_dict.values():
                total_params += v.numel()

            unexpected_params = 0
            for k in unexpected:
                unexpected_params += state_dict[k].numel()

            used_params = total_params - unexpected_params

            if len(unexpected) > 0:
                logger.warning(f"OnlyImagingModule: {unexpected_params} unexpected params when loading pretrained vision encoder. Unexpected keys in state_dict when loading pretrained vision encoder from {pretrained_vlp_module}: {unexpected}.")

            if len(missing) > 0:
                logger.debug(f"OnlyImagingModule: {len(missing)} missing keys when loading pretrained vision encoder. This is expected for the missing classification head.")

            logger.info(f"OnlyImagingModule: Loaded pretrained vision encoder with {used_params} parameters from {pretrained_vlp_module}")
            

        self.label_weights = torch.Tensor(label_weights) # first element gives the weight for class 0 and the second element gives the weight for class 1
        logger.debug(f"OnlyImagingModule: label_weights: {self.label_weights} for class 0 and 1, respectively")

        self._create_metrics_and_caches()

        logger.info(f"OnlyImagingModule: Successfully initialized {model} model with the hyperparameters: {self.hparams}")

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
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
    
    def setup(self, stage):
        logger.debug(f"OnlyImagingModule: Setup: Sending everything to the device: {self.device}")
        # Interestingly is the self.device not correctly set in the __init__ method, so we need to set it here
        self.network = self.network.to(self.device)
        self.label_weights = self.label_weights.to(self.device)
        
        self._move_metrics_and_caches_to_device()

    def _create_metrics_and_caches(self):
        """
        Initialize binary classification metrics for training, validation and data caches for validation.
        
        Creates separate metric instances for different phases (train/val) and datasets 
        (internal/btxrd/combined). Also initializes tensor caches for storing predictions, 
        labels, logits, features, and dataset labels during validation epochs.
        
        Note:
            PyTorch Lightning requires metrics to be created as class attributes rather than
            directly in dictionaries for proper state management.
        
        Sets up:
            - Training metrics: accuracy, precision, recall, f1, auroc
            - Validation metrics: separate instances for internal, btxrd, and combined datasets
            - Test metrics: separate instances for internal, btxrd, and combined datasets  
            - Data caches: tensors for storing validation predictions, labels, 
              logits, features, and dataset labels for epoch-end evaluation
        """

        self.train_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_auroc = BinaryAUROC()

        self.val_internal_accuracy = BinaryAccuracy()
        self.val_internal_precision = BinaryPrecision()
        self.val_internal_recall = BinaryRecall()
        self.val_internal_f1 = BinaryF1Score()
        self.val_internal_auroc = BinaryAUROC()
        self.val_btxrd_accuracy = BinaryAccuracy()
        self.val_btxrd_precision = BinaryPrecision()
        self.val_btxrd_recall = BinaryRecall()
        self.val_btxrd_f1 = BinaryF1Score()
        self.val_btxrd_auroc = BinaryAUROC()
        self.val_combined_accuracy = BinaryAccuracy()
        self.val_combined_precision = BinaryPrecision()
        self.val_combined_recall = BinaryRecall()
        self.val_combined_f1 = BinaryF1Score()
        self.val_combined_auroc = BinaryAUROC()

        self.train_metrics = {
            "accuracy": self.train_accuracy,
            "precision": self.train_precision,
            "recall": self.train_recall,
            "f1": self.train_f1,
            "auroc": self.train_auroc,
        }

        # seperate metrics for validation datasets
        self.val_metrics = {
            "internal": {
                "accuracy": self.val_internal_accuracy,
                "precision": self.val_internal_precision,
                "recall": self.val_internal_recall,
                "f1": self.val_internal_f1,
                "auroc": self.val_internal_auroc,
            },
            "btxrd": {
                "accuracy": self.val_btxrd_accuracy,
                "precision": self.val_btxrd_precision,
                "recall": self.val_btxrd_recall,
                "f1": self.val_btxrd_f1,
                "auroc": self.val_btxrd_auroc,
            },
            "combined": {
                "accuracy": self.val_combined_accuracy,
                "precision": self.val_combined_precision,
                "recall": self.val_combined_recall,
                "f1": self.val_combined_f1,
                "auroc": self.val_combined_auroc,
            },
        }

        # storing for combined validation evaluation for on_validation_epoch_end
        self.all_val_probs = torch.empty(0)
        self.all_val_labels = torch.empty(0)
        self.all_val_logits = torch.empty(0)
        self.all_val_features = torch.empty(0)  # features for the coral loss
        self.all_val_datset_labels = [] # storing the dataset labels for the coral loss


    def _move_metrics_and_caches_to_device(self):
        """
        Moves all metrics and data caches to the specified device.
        
        Note:
            This needed to be in a separate method since in the __init__ method the self.device is not correctly set yet.
        """

        # Move train metrics
        for metric in self.train_metrics.values():
            metric.to(self.device)
        
        # Move val metrics
        for _, metrics in self.val_metrics.items():
            for metric in metrics.values():
                metric.to(self.device)

        self.all_val_probs = self.all_val_probs.to(self.device)
        self.all_val_labels = self.all_val_labels.to(self.device)
        self.all_val_logits = self.all_val_logits.to(self.device)
        self.all_val_features = self.all_val_features.to(self.device)


    def forward(self, x):
        return self.network(x).flatten()
    

    def get_image_network(self):
        return self.network
    
    # To capture features before being passed to the classification head, we follow the pattern of timm models and also expose forward_features and forward_head methods
    # Extracted features are needed for coral loss computation and can be used for t-SNE plotting and silhouette score calculation
    def forward_features(self, x):
        return self.network.forward_features(x)
    
    def forward_head(self, x):
        return self.network.forward_head(x).flatten()
    
    def _compute_loss(self, features, logits, labels, dataset):
        """
        Compute the combined loss including classification loss and optional CORAL domain adaptation loss.
        Args:
            features (torch.Tensor): Feature representations from the model. Can be either 2D tensor of shape 
                                   (batch_size, num_features) or 4D tensor of shape (batch_size, num_features, height, width).
            logits (torch.Tensor): Raw model predictions before sigmoid activation of shape (batch_size,).
            labels (torch.Tensor): Ground truth binary labels of shape (batch_size,).
            dataset (list): List of strings indicating the dataset source for each sample ("INTERNAL" or "BTXRD").
        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): Combined classification and CORAL loss (or just classification loss if CORAL is disabled/unavailable).
                - classification_loss (torch.Tensor): Binary cross-entropy loss with class weights.
                - coral_loss (torch.Tensor): CORAL domain adaptation loss weighted by lambda parameter, or 0.0 if disabled/unavailable.
        Notes:
            - Uses weighted binary cross-entropy loss with per-class weights defined in self.label_weights.
            - CORAL loss is only computed if self.hparams.coral_lambda > 0.0 and there are at least 2 samples
              from each dataset (INTERNAL and BTXRD) in the batch.
            - For 4D feature tensors, applies average pooling over spatial dimensions before CORAL computation.
            - Returns zero CORAL loss if conditions for computation are not met (insufficient samples per domain).
        """
        ### BCE Loss
        # BCEWithLogitsLoss expects for the weight argument the weight on a per sample basis
        # So, we need to create this weight vector depending on the labels
        sample_weights = torch.where(labels == 0, self.label_weights[0], self.label_weights[1])
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), weight=sample_weights)

        ### CORAL Loss
        if self.hparams.coral_lambda == 0.0: # if lambda is 0.0, we skip the coral loss calculation
            return classification_loss, classification_loss, torch.tensor(0.0, device=self.device)
        
        if len(features.shape) == 4: # avg pool if the features are of shape (batch_size, num_features, height, width)
            features_avg_pooled = torch.mean(features, dim=(2, 3)) # coral loss expects the features to be of shape (batch_size, num_features)
        else:
            features_avg_pooled = features

        # get the features corresponding to the dataset to have them seperate for the coral loss calculation
        internal_mask = torch.tensor([d == "INTERNAL" for d in dataset], dtype=torch.bool)
        btxrd_mask = torch.tensor([d == "BTXRD" for d in dataset], dtype=torch.bool)

        if sum(internal_mask) <= 1 or sum(btxrd_mask) <= 1: # for at least one dataset there is at most one sample in the batch, the coral loss gets NaN if only one sample is present for a dataset
            return classification_loss, classification_loss, torch.tensor(0.0, device=self.device) # if there are only samples from one dataset, we cannot compute the coral loss, so we return 0.0 for the coral loss
        else:
            features_internal = features_avg_pooled[internal_mask]
            features_btxrd = features_avg_pooled[btxrd_mask]
            coral_loss = coral(features_internal, features_btxrd)
            # weight the coral loss by the lambda parameter
            coral_loss = self.hparams.coral_lambda * coral_loss

            loss = classification_loss + coral_loss

            return loss, classification_loss, coral_loss


    def training_step(self, batch, batch_idx):
        x, labels, dataset = batch["x-ray"], batch["tumor"], batch["dataset"]
        features = self.forward_features(x)
        logits: torch.Tensor = self.forward_head(features)

        loss, classification_loss, coral_loss = self._compute_loss(features, logits, labels, dataset)

        probs = torch.sigmoid(logits)

        # Binary* metrics handle thresholding and calcuating predictions from probs internally
        train_metrics = self.train_metrics
        train_metrics['accuracy'](probs, labels)
        train_metrics['precision'](probs, labels)
        train_metrics['recall'](probs, labels)
        train_metrics['f1'](probs, labels)
        train_metrics['auroc'](probs, labels)


        batch_size = x.shape[0]
        # log the losses on every step
        self.log("train/classification_loss", classification_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/coral_loss", coral_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        # but only log aggregated metrics on epoch
        self.log("train/accuracy", train_metrics["accuracy"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/precision", train_metrics["precision"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/recall", train_metrics["recall"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/f1", train_metrics["f1"], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/auroc", train_metrics["auroc"], on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, labels, dataset = batch["x-ray"], batch["tumor"], batch["dataset"]
        features = self.forward_features(x)
        logits: torch.Tensor = self.forward_head(features)

        loss, _, _ = self._compute_loss(features, logits, labels, dataset)

        probs = torch.sigmoid(logits)

        # save all probs, labels and logits for combined validation metrics in on_validation_epoch_end
        self.all_val_probs = torch.cat([self.all_val_probs, probs], dim=0)
        self.all_val_labels = torch.cat([self.all_val_labels, labels], dim=0)
        self.all_val_logits = torch.cat([self.all_val_logits, logits], dim=0)
        self.all_val_features = torch.cat([self.all_val_features, features], dim=0)  # features for the coral loss
        self.all_val_datset_labels.extend(dataset)  # store the dataset labels for the coral loss

        # We assume that the first dataloader is for the INTERNAL dataset and the second dataloader for the BTXRD dataset
        if dataloader_idx == 0:
            key = "internal"
            log_path_str = "val/internal"
        elif dataloader_idx == 1:
            key = "btxrd"
            log_path_str = "val/btxrd"
        else:
            raise ValueError(
                f"OnlyImagingModule: Validation dataloader index {dataloader_idx} is not supported. Supported indices are: 0, 1. We are assuming that the first dataloader is for the INTERNAL dataset and the second dataloader for the BTXRD dataset"
            )

        metrics = self.val_metrics[key]

        # Binary* metrics handle thresholding and calcuating predictions from probs internally
        metrics['accuracy'](probs, labels)
        metrics['precision'](probs, labels)
        metrics['recall'](probs, labels)
        metrics['f1'](probs, labels)
        metrics['auroc'](probs, labels)

        batch_size = x.shape[0]
        # log the loss on every step and epoch
        self.log(f"{log_path_str}/loss", loss, on_step=True, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        # but only log aggregated metrics on epoch
        self.log(f"{log_path_str}/accuracy", metrics["accuracy"], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/precision", metrics["precision"], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/recall", metrics["recall"], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/f1", metrics["f1"], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/auroc", metrics["auroc"], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)

        return loss
    
    def on_validation_epoch_end(self):
        """
        Compute and log validation metrics on the combined validation set.
        
        This function is used since the regular validation step computes metrics
        on internal and btxrd datasets separately due to separate dataloaders given, but we also want to evaluate
        performance on the combined validation set to get an overall picture of
        model performance across all validation data.
        
        The function:
        1. Computes total loss (classification + coral) on accumulated validation data
        2. Calculates classification metrics (accuracy, precision, recall, f1, auroc)
        3. Logs all metrics with 'val/combined/' prefix for tracking
        4. Clears accumulated validation data and resets metric states for next epoch
        """
        loss, classification_loss, coral_loss = self._compute_loss(self.all_val_features, self.all_val_logits, self.all_val_labels, self.all_val_datset_labels)

        metrics = self.val_metrics['combined']

        accuracy = metrics['accuracy'](self.all_val_probs, self.all_val_labels)
        precision = metrics['precision'](self.all_val_probs, self.all_val_labels)
        recall = metrics['recall'](self.all_val_probs, self.all_val_labels)
        f1 = metrics['f1'](self.all_val_probs, self.all_val_labels)
        auroc = metrics['auroc'](self.all_val_probs, self.all_val_labels)

        batch_size = self.all_val_features.shape[0]
        self.log("val/combined/loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/classification_loss", classification_loss, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/coral_loss", coral_loss, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/accuracy", accuracy, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/precision", precision, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/recall", recall, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/f1", f1, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log("val/combined/auroc", auroc, on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)

        # Clear cache
        self.all_val_probs = torch.empty(0, device=self.device)
        self.all_val_labels = torch.empty(0, device=self.device)
        self.all_val_logits = torch.empty(0, device=self.device)
        self.all_val_features = torch.empty(0, device=self.device)  # features for the coral loss
        self.all_val_datset_labels = []

        # manually clear combined metrics, since this is not done automatically by lightning, since this is not the regular validation step
        for _, m in metrics.items():
            m.reset()


if __name__ == "__main__":
    # Dummy input
    x = torch.randn(3, 3, 224, 224)  # Batch size of 1, 1 channels, 224x224 image
    y = torch.tensor([1, 1, 0])  # Binary labels
    dataset = ["INTERNAL", "BTXRD", "INTERNAL"]  # Dummy dataset labels

    # Example usage
    w0 = len(y) / (2 * torch.sum(y == 0))
    w1 = len(y) / (2 * torch.sum(y == 1))
    label_weights = (w0, w1)
    model = OnlyImagingModule(model="resnet34", optimizer=torch.optim.Adam, label_weights=label_weights)
    
    print(f"Predicted {torch.sigmoid(model(x))}")
    print(f"Actual {y}")
    model.training_step({"x-ray": x, "tumor": y, "dataset": dataset}, 0)
