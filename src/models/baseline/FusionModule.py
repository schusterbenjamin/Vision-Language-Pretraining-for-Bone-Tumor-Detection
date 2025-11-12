import os
import random
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


class FusionModule(L.LightningModule):
    def __init__(
        self,
        model: str, # TODO: allow it to be a timm model already and handle this case, this is then the support of using a pretrained vision encoder from the VLP. I also need to replace the classifier layers.
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        label_weights: Tuple[float] = (1.0, 1.0), # this will be set by the train script and should NOT be configured in the hydra config!
        coral_lambda: float = 0.0,  # lambda for the coral loss
        pretrained_vlp_module: str = None, # if a pretrained vision language module is provided, we use its vision encoder as pretrained image model
        vision_encoder_lr: float = None, # if provided, we use this learning rate for the vision encoder, otherwise the learning rate of the optimizer is used
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False) # the hyperparameters are already logged by the train script including also data, trainer, etc.

        assert vision_encoder_lr is None or vision_encoder_lr >= 0.0, "FusionModule: vision_encoder_lr must be None or >= 0.0"

        if model not in supported_models:
            raise ValueError(
                f"FusionModule: Model {model} is not supported. Supported models are: {supported_models}"
            )
        
        # tabular clinical data network
        self.tabular_network = torch.nn.Sequential(
            torch.nn.Linear(15, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
        )

         # catch special case for second baseline: resnet50 pretrained on chest x-rays from torchxrayvision
        if model == "resnet50-res512-all":
            class PretrainedResnet(torch.nn.Module):
                def __init__(self):
                    super(PretrainedResnet, self).__init__()
                    self.vision_encoder = xrv.models.ResNet(weights="resnet50-res512-all", cache_dir='./.cache/torchxrayvision/')
                    self.classififer = torch.nn.Linear(in_features=2048, out_features=10)
                def forward(self, x):
                    return self.classififer(self.vision_encoder.features(x))
                def forward_features(self, x):
                    return self.vision_encoder.features(x)
                def forward_head(self, x):
                    return self.classififer(x)
                
            self.image_network = PretrainedResnet()
            logger.info("FusionModule: Using resnet50-res512-all from torchxrayvision as image model.")
        # if no pretrained vision language module is provided, train from scratch
        elif pretrained_vlp_module is None:
            # We do binary classification, so we set num_classes=1
            self.image_network = timm.create_model(model, num_classes=10)
        else:
            checkpoint = torch.load(pretrained_vlp_module, map_location="cpu", weights_only=False)
            # get all key value pairst that start with "image_encoder.model." and remove the rest
            state_dict = {k.replace("image_encoder.model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("image_encoder.model.")}
            self.image_network = timm.create_model(model, num_classes=10, pretrained=False)
            missing, unexpected = self.image_network.load_state_dict(state_dict, strict=False)
             # get how many params have been loaded
            total_params = 0
            for v in state_dict.values():
                total_params += v.numel()

            unexpected_params = 0
            for k in unexpected:
                unexpected_params += state_dict[k].numel()

            used_params = total_params - unexpected_params

            if len(unexpected) > 0:
                logger.warning(f"FusionModule: {unexpected_params} unexpected params when loading pretrained vision encoder. Unexpected keys in state_dict when loading pretrained vision encoder from {pretrained_vlp_module}: {unexpected}.")

            if len(missing) > 0:
                logger.debug(f"FusionModule: {len(missing)} missing keys when loading pretrained vision encoder. This is expected for the missing classification head.")

            logger.info(f"FusionModule: Loaded pretrained vision encoder with {used_params} parameters from {pretrained_vlp_module}")

        # combination network
        self.combination_network = torch.nn.Linear(20, 1)  # 10 features from the image network and 10 features from the tabular network

        self.label_weights = torch.Tensor(label_weights) # first element gives the weight for class 0 and the second element gives the weight for class 1
        logger.debug(f"FusionModule: label_weights: {self.label_weights} for class 0 and 1, respectively")

        self._create_metrics_and_caches()

        logger.info(f"FusionModule: Successfully initialized {model} model with the hyperparameters: {self.hparams}")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for the fusion module.
        This method sets up different parameter groups with potentially different learning rates:
        - If vision_encoder_lr is specified and >= 0.0, creates separate parameter groups for:
          * Image backbone parameters (features) with vision_encoder_lr
          * Image head parameters (head/classifier/fc layers) and remaining model parameters with default lr
        - Otherwise, uses default learning rate for all parameters
        The method identifies image head parameters by checking if parameter names contain
        'head', 'classifier', or 'fc'. All other image network parameters are considered
        backbone parameters.
        Returns:
            dict: Dictionary containing optimizer configuration. If scheduler is specified,
                  also includes lr_scheduler configuration with epoch-based scheduling.
        Logs:
            - Parameter counts for each group
            - Learning rates being used
            - Total parameters being optimized
        """
        param_groups = []
        if self.hparams.vision_encoder_lr is not None and self.hparams.vision_encoder_lr >= 0.0:
            # Split image network into backbone (features) and head
            image_backbone_params = []
            image_head_params = []
            
            # Get all parameters and identify head vs backbone
            for name, param in self.image_network.named_parameters():
                if 'head' in name or 'classifier' in name or 'fc' in name:
                    logger.info(f"FusionModule: Adding parameter {name} to image head parameters")
                    image_head_params.append(param)
                else:
                    image_backbone_params.append(param)
            
            # Get parameter IDs for remaining network components
            image_network_params_ids = set(id(p) for p in self.image_network.parameters())
            remaining_params = [p for p in self.parameters() if id(p) not in image_network_params_ids]
            
            # Combine head parameters with remaining parameters
            head_and_remaining_params = image_head_params + remaining_params
            
            # Count parameters in each group
            image_backbone_param_count = sum(p.numel() for p in image_backbone_params)
            head_and_remaining_param_count = sum(p.numel() for p in head_and_remaining_params)
            
            param_groups.append({"params": image_backbone_params, "lr": self.hparams.vision_encoder_lr, "name": "image_backbone"})
            param_groups.append({"params": head_and_remaining_params, "name": "head_and_remaining_parameters"})
            
            optimizer = self.hparams.optimizer(params=param_groups)        
        
            logger.info(f"FusionModule: Using separate learning rate {self.hparams.vision_encoder_lr} for the vision encoder backbone with {image_backbone_param_count:,} parameters")
            logger.info(f"FusionModule: Using learning rate {optimizer.defaults['lr']} for the head and remaining parameters with {head_and_remaining_param_count:,} parameters")
            
            # Log the total parameters being optimized
            total_params = image_backbone_param_count + head_and_remaining_param_count
            logger.info(f"FusionModule: Total parameters being optimized: {total_params:,}")
        else:
            total_params = sum(p.numel() for p in self.parameters())
            optimizer = self.hparams.optimizer(params=self.parameters())
            logger.info(f"FusionModule: Using default learning rate {optimizer.defaults['lr']} for all {total_params:,} parameters")

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
        logger.debug(f"FusionModule: Setup: Sending everything to the device: {self.device}")
        # Interestingly is the self.device not correctly set in the __init__ method, so we need to set it here
        self.network = self.image_network.to(self.device)
        self.tabular_network = self.tabular_network.to(self.device)
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
        self.all_val_image_features = torch.empty(0)  # features for the coral loss
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
        self.all_val_image_features = self.all_val_image_features.to(self.device)
    

    def get_image_network(self):
        return self.image_network

    def forward(self, x, age_encoded, sex_encoded, anatomy_site_encoded):
        """
        This also passes the image_features s.t. the image features can be used for the coral loss in the training step
        If you dont need the image features, please get rid of them when calling this function.
        """
        image_features = self.forward_image_features(x)
        image_logits: torch.Tensor = self.forward_image_head(image_features)

        clinical_data_combined = torch.cat((anatomy_site_encoded, age_encoded, sex_encoded), dim=1)
        clinical_logits = self.tabular_network(clinical_data_combined)

        logits: torch.Tensor = self.combination_network(torch.cat((image_logits, clinical_logits), dim=1)).flatten()

        return logits, image_features
    
    # To capture features before being passed to the classification head, we follow the pattern of timm models and also expose forward_features and forward_head methods
    # Extracted features are needed for coral loss computation and can be used for t-SNE plotting and silhouette score calculation
    def forward_image_features(self, x):
        return self.image_network.forward_features(x)
    
    def forward_image_head(self, x):
        return self.image_network.forward_head(x)
    
    def _compute_loss(self, image_features, logits, labels, dataset):
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
        if self.hparams.coral_lambda == 0.0:
            return classification_loss, classification_loss, torch.tensor(0.0, device=self.device)
        # get the features corresponding to the dataset to have them seperate for the coral loss calculation
        if len(image_features.shape) == 4: # only avg pool, if needed # if the features are of shape (batch_size, num_features, height, width)
            features_avg_pooled = torch.mean(image_features, dim=(2, 3)) # coral loss expects the features to be of shape (batch_size, num_features)
        else:
            features_avg_pooled = image_features
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
        anatomy_site, age, sex = batch["anatomy_site_encoded"], batch["age_encoded"], batch["sex_encoded"]
        
        logits, image_features = self.forward(x, age, sex, anatomy_site)
        loss, classification_loss, coral_loss = self._compute_loss(image_features, logits, labels, dataset)
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
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, labels, dataset = batch["x-ray"], batch["tumor"], batch["dataset"]
        anatomy_site, age, sex = batch["anatomy_site_encoded"], batch["age_encoded"], batch["sex_encoded"]
        logits, image_features = self.forward(x, age, sex, anatomy_site)
        loss, _, _ = self._compute_loss(image_features, logits, labels, dataset)
        probs = torch.sigmoid(logits)

        # save all probs, labels and logits for combined validation metrics in on_validation_epoch_end
        self.all_val_probs = torch.cat([self.all_val_probs, probs], dim=0)
        self.all_val_labels = torch.cat([self.all_val_labels, labels], dim=0)
        self.all_val_logits = torch.cat([self.all_val_logits, logits], dim=0)
        self.all_val_image_features = torch.cat([self.all_val_image_features, image_features], dim=0)  # features for the coral loss
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
                f"FusionModule: Validation dataloader index {dataloader_idx} is not supported. Supported indices are: 0, 1. We are assuming that the first dataloader is for the INTERNAL dataset and the second dataloader for the BTXRD dataset"
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
        self.log(f"{log_path_str}/accuracy", metrics['accuracy'], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/precision", metrics['precision'], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/recall", metrics['recall'], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/f1", metrics['f1'], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{log_path_str}/auroc", metrics['auroc'], on_step=False, on_epoch=True, add_dataloader_idx=False, batch_size=batch_size)

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

        loss, classification_loss, coral_loss = self._compute_loss(self.all_val_image_features, self.all_val_logits, self.all_val_labels, self.all_val_datset_labels)

        metrics = self.val_metrics['combined']

        accuracy = metrics['accuracy'](self.all_val_probs, self.all_val_labels)
        precision = metrics['precision'](self.all_val_probs, self.all_val_labels)
        recall = metrics['recall'](self.all_val_probs, self.all_val_labels)
        f1 = metrics['f1'](self.all_val_probs, self.all_val_labels)
        auroc = metrics['auroc'](self.all_val_probs, self.all_val_labels)

        batch_size = self.all_val_image_features.shape[0]
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
        self.all_val_image_features = torch.empty(0, device=self.device)  # features for the coral loss
        self.all_val_datset_labels = []

        # manually clear combined metrics
        for _, m in metrics.items():
            m.reset()


if __name__ == "__main__":
    # Dummy input
    batch_size = 30
    x = torch.randn(batch_size, 3, 224, 224)  # Batch size of 1, 1 channels, 224x224 image
    y = torch.tensor([random.randint(0, 1) for _ in range(batch_size)])  # Binary labels
    dataset = [random.choice(["INTERNAL", "BTXRD"]) for _ in range(batch_size)]  # Dummy dataset labels
    anatomy_site_encoded = torch.zeros(batch_size, 13)  # Dummy encoded anatomy site
    random_indices = [random.randint(0, 12) for _ in range(batch_size)]
    anatomy_site_encoded[torch.arange(batch_size), random_indices] = 1  # One-hot encoding
    age_encoded = torch.randint(1, 8, (batch_size, 1))
    sex_encoded = torch.randint(0, 2, (batch_size, 1))

    # Example usage
    w0 = len(y) / (2 * torch.sum(y == 0))
    w1 = len(y) / (2 * torch.sum(y == 1))
    label_weights = (w0, w1)
    model = FusionModule(model="resnet34", optimizer=torch.optim.Adam, label_weights=label_weights, coral_lambda=100)
    
    print(f"Predicted {torch.sigmoid(model(x, age_encoded, sex_encoded, anatomy_site_encoded)[0])}")
    print(f"Actual {y}")
    model.training_step({"x-ray": x, "tumor": y, "dataset": dataset, "age_encoded": age_encoded, "sex_encoded": sex_encoded, "anatomy_site_encoded": anatomy_site_encoded}, 0)