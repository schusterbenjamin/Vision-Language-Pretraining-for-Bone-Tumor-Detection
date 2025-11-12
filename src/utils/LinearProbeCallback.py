import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import torch
import lightning as pl
from lightning import Callback
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pretrain.VisionLanguageModule import VisionLanguageModule

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('project')


class LinearProbeCallback(Callback):
    """
    A PyTorch Lightning Callback that trains a linear probe on the image encoder's features
    at the start of every n-th validation epoch and logs the balanced accuracy and AUROC.
    """
    def __init__(self, train_dataloader: DataLoader, val_dataloaders: list[DataLoader], every_n_epochs: int = 5):
        super().__init__()
        self.train_dataloader = train_dataloader

        val_datasets = [dl.dataset for dl in val_dataloaders]
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        self.val_dataloader = DataLoader(val_dataset, batch_size=train_dataloader.batch_size, shuffle=False)

        self.every_n_epochs = every_n_epochs

    # having it on validation start for two reasons: 1) I can log with the pl_module 2) it will be available for the early stopping callback
    def on_validation_start(self, trainer: pl.Trainer, pl_module: VisionLanguageModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        
        if trainer.sanity_checking:
            # Skip if we are in the sanity check phase
            return
        
        image_encoder = pl_module.image_encoder
        logger.debug(f"LinearProbeCallback: Image Encoder in mode: {image_encoder.training} (True means training, False means eval)")
        
        balanced_accuracy, auroc = self._linear_probe_training(image_encoder, pl_module.device)

        pl_module.log("downstream_validation/linear_probe_balanced_accuracy", balanced_accuracy, on_step=False, on_epoch=True)
        pl_module.log("downstream_validation/linear_probe_auroc", auroc, on_step=False, on_epoch=True)

        # trainer.logger.log_metrics({"downstream_validation/linear_probe_balanced_accuracy": balanced_accuracy}, step=trainer.global_step)
        # trainer.logger.log_metrics({"downstream_validation/linear_probe_auroc": auroc}, step=trainer.global_step)
        # logger.info(f"LinearProbeCallback: Linear probe accuracy: {balanced_accuracy}")
        
    def _linear_probe_training(self, image_encoder, device):
        """
        Train a linear probe on the image encoder's features using Logistic Regression.
        Returns the balanced accuracy and AUROC on the validation set.

        Args:
            image_encoder (nn.Module): The image encoder model.
            device (str): The device to run the computations on.

        Returns:
            (balanced_accuracy, auroc) (float, float): Tuple of balanced accuracy and AUROC score.
        """
        image_encoder = image_encoder.eval()

        logger.debug(f"LinearProbeCallback: Extracting features for linear probe training")
        X_train, y_train = self._extract_features(image_encoder, self.train_dataloader, device)
        X_val, y_val = self._extract_features(image_encoder, self.val_dataloader, device)
        logger.debug("LinearProbeCallback: Extracted features for linear probe training")

        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
        )
        logger.debug(f"LinearProbeCallback: Training linear probe using LogisticRegression")
        logger.debug(f"LinearProbeCallback: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        clf.fit(X_train, y_train)
        logger.debug(f"LinearProbeCallback: Evaluating linear probe")

        y_pred = clf.predict(X_val)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)

        y_scores = clf.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_scores)

        # logger.debug(f"LinearProbeCallback: Linear probe accuracy: {acc}")

        return balanced_acc, auroc
        

    def _extract_features(self, encoder, dataloader, device):
        """
        Extract features from the encoder for all samples in the dataloader.
        
        Args:
            encoder (nn.Module): The image encoder model.
            dataloader (DataLoader): DataLoader for the dataset.
            device (str): The device to run the computations on.
            
        Returns:
            (X, y) (np.ndarray, np.ndarray): Extracted features, Corresponding labels
        """
        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features for linear probe"):
                imgs, labels = batch["x-ray"], batch["tumor"]
                imgs = imgs.to(device)
                feats = encoder(imgs)           # [B, D]
                feats = feats.cpu().numpy()
                all_feats.append(feats)
                all_labels.append(labels.cpu().numpy())

        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)
        return X, y
        