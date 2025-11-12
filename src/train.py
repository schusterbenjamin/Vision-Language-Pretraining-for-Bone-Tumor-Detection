from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import rootutils
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb





rootutils.setup_root(__file__, indicator=".env", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.models.pretrain.VisionLanguageModule import VisionLanguageModule
from src.data.KFoldCVDataModule import KFoldCVDataModule
from src.models.baseline.OnlyImagingModule import OnlyImagingModule
from src.models.baseline.FusionModule import FusionModule
# I need to add this import here s.t. hydra can find the DownstreamDataModule class (WTF :D)
from src.data.DownstreamDataModule import DownstreamDataModule
from src.utils.LinearProbeCallback import LinearProbeCallback
from src.data.PretrainDataModule import PretrainDataModule
from src.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from src.utils.hyperparam_logging import log_hyperparameters
from utils.plotting.plot_tsne_and_calculate_silhouette import plot_train_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule, plot_validation_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule
from src.utils.plotting.plot_confusion_matrix import plot_validation_confusion_matrix_from_own_model_and_datamodule, plot_train_confusion_matrix_from_own_model_and_datamodule

import logging
logging.config.fileConfig('logging.conf')
log = logging.getLogger('project')


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains a model. Highly configurable via Hydra config files.

    Executes the following steps:
    1. Instantiates a KFoldCVDataModule for k-fold cross validation.
    2. For each fold:
        a. Instantiates the model.
        b. Instantiates callbacks and loggers.
        c. If if model is VisionLanguageModule and downstream data is provided: Instantiates a DownstreamDataModule and adds LinearProbeCallback.
        d. Instantiates the Trainer.
        e. Logs hyperparameters.
        f. Trains the model.
        g. If model is baseline module: Plots t-SNE and confusion matrix for train and val data.
        h. If model is VisionLanguageModule and downstream data is provided: Evaluates model on downstream task in zero-shot setting.
    3. If k-fold cross validation is used: Aggregates metrics across folds and logs them.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # boolean indicating if we are using k-fold cross validation
    # If we are not using k-fold cross validation, we still use a KFoldCVDataModule but just train on the first fold
    k_fold_cross_validation = cfg.get("k_fold_cross_validation", False)
    if k_fold_cross_validation:
        log.info("Train: Doing k-fold cross validation.")
    else:
        log.info("Train: Not doing k-fold cross validation. Using first fold only.")

    cross_validation_group = wandb.util.generate_id() # create a  group name to bundle k-fold cv runs together in wandb
    
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Train: Instantiating datamodule <{cfg.data._target_}>")
    datamodule: KFoldCVDataModule = hydra.utils.instantiate(cfg.data)

    # if there is a wandb logger, save the run name for later usage
    wandb_used = False
    logger_config = cfg.get("logger")
    if logger_config and isinstance(logger_config, DictConfig):
        for _, lg_conf in logger_config.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                if "WandbLogger" in lg_conf._target_:
                    wandb_run_name = lg_conf['name']
                    wandb_used = True
                    break  # there shouldnt be more than one wandb logger anyways
    
    all_fold_metrics = []
    for i, (k_fold_datamodule, label_weights) in enumerate(datamodule.get_cv_splits()):
        log.info(f"Starting initializationg for fold {i}")

        log.info(f"Train: Instantiating model <{cfg.model._target_}>")
        cfg.model = OmegaConf.to_container(cfg.model, resolve=True)
        # if model.scheduler is an empty dict, set it to None, this is needed s.t. sweeps can also use the no_scheduler configuration (which is an empty dict)
        if not cfg.model.get("scheduler"):
            cfg.model["scheduler"] = None
        # set the pos_weight in the model config
        with open_dict(cfg.model): # this disables the struct mode of the config (needed in order to set a new key, that was not in the original config)
            cfg.model["label_weights"] = label_weights
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        if isinstance(model, VisionLanguageModule):
            if not isinstance(datamodule, PretrainDataModule):
                log.warning("Train: Using a VisionLanguageModule but the datamodule is not a PretrainDataModule. This might lead to errors during training.")

        log.info("Train: Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        # for a VisionLanguageModule, we can define the downstream dataset to be used for Linear Probing during pretraining
        if cfg.get("downstream_data"):
            downstream_datamodule: DownstreamDataModule = hydra.utils.instantiate(cfg.downstream_data)
            dm, _ = next(downstream_datamodule.get_cv_splits()) # get the first split of k-fold cv
            downstream_train_dataloader = dm.train_dataloader()
            downstream_val_dataloaders = dm.val_dataloader()
            # if there is a downstream datamodule defined and the model is a VisionLanguageModule, then we want to apply LinearProbing 
            if isinstance(model, VisionLanguageModule):
                linear_probe_callback = LinearProbeCallback(downstream_train_dataloader, downstream_val_dataloaders)
                callbacks.append(linear_probe_callback)
                log.info("Train: Added LinearProbeCallback to callbacks.")

        log.info("Train: Instantiating loggers...")
        # if there is a wandb logger, add the group to it and append the fold number to the name
        with open_dict(cfg.logger): # this disables the struct mode of the config (needed in order to set a new key, that was not in the original config)
            if logger_config and isinstance(logger_config, DictConfig):
                for _, lg_conf in logger_config.items():
                    if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                        if "WandbLogger" in lg_conf._target_:
                            if k_fold_cross_validation:
                                lg_conf['group'] = cross_validation_group
                                lg_conf['name'] = wandb_run_name + f"_fold:{i}"
                            else:
                                lg_conf['name'] = wandb_run_name
                            break  # there shouldnt be more than one wandb logger anyways

        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

        log.info(f"Train: Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Train: Logging hyperparameters!")
            log_hyperparameters(object_dict)

        if cfg.get("train"):
            log.info("Train: Starting training!")
            trainer.fit(model=model, datamodule=k_fold_datamodule, ckpt_path=cfg.get("ckpt_path"))

        if wandb_used: # if there is a wandb logger, use its summary metrics for the fold (they should be set by the SnapshotAllMetricsOnBestCallback)
            fold_metrics = dict(wandb.run.summary)
        else: # otherwise use the trainer callback metrics, BUT these are just the last metrics and not the ones from the best model
            fold_metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
        all_fold_metrics.append(fold_metrics)

        # Plot tsne and confusion matrix of validation and train data, if it is a baseline module
        if isinstance(model, (OnlyImagingModule, FusionModule)):
            log.info("Train: Plotting t-SNE and confusion matrix and calculating silhouette score of train and validation data...")
            # if wandb is used, we can log the plots
            plot_and_wandb_log_train_and_val_tsne_and_confusion_matrix_and_calculate_silhouette_score(model, k_fold_datamodule, wandb_used, trainer)
        else:
            log.info("Train: Not plotting t-SNE and confusion matrix and calculating silhouette score of train and validation data, since model is not a baseline module.")

        if isinstance(model, VisionLanguageModule):
            if cfg.get("downstream_data"):
                if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                    log.info(f"Train: Using best model path from trainer: {trainer.checkpoint_callback.best_model_path}")
                    best_model_path = trainer.checkpoint_callback.best_model_path

                    # the downstream datamodule is not saved during checkpointing, so we need to instantiate it again
                    # simply setting it to module.downstream_datamodule will cause the VisionLanguageModule to use a different split,
                    # since the get_cv_splits method of the KFoldCVDataModule will return a different split each time it is called (it uses yield)
                    # it is not pretty, since we load the DownstreamDataModule twice in total, but it is quite fast, so it is worth having a bit cleaner setup
                    downstream_datamodule: DownstreamDataModule = hydra.utils.instantiate(cfg.downstream_data)
                    model = type(model).load_from_checkpoint(best_model_path, downstream_datamodule=downstream_datamodule)
                    # model.set_tokenization_function(datamodule.tokenize_captions) # we also need to set the tokenization function again, since the downstream datamodule is not saved in the checkpoint
                else:
                    log.warning("Train: No best model path found in trainer, using current model weights. Please use a ModelCheckpoint callback to save the best model during training.")
                
                # log.info("Train: Evaluating model performance and precision at k on the downstream task in zero-shot setting...")
                downstream_precision_at_k = model.evaluate_downstream_precision_at_k(mode='entire')
                if downstream_precision_at_k:
                    for k, v in downstream_precision_at_k.items():
                        wandb.log({f"downstream_entire/label_precision_at_{k}": v})
                # metrics, downstream_precision_at_k = model.evaluate_downstream_zero_shot_and_precision_at_k(mode='entire')
                # wandb.log({f"downstream_entire/zero_shot/{k}": v for k, v in metrics.items()})
                # wandb.log({f"downstream_entire/label_precision_at_{k}": v for k, v in downstream_precision_at_k.items()})
                log.info("Train: Finished evaluating model performance and precision at k on the downstream task in zero-shot setting.")
            else:
                log.warning("Train: No downstream data provided, skipping evaluation on the downstream task.")
        else:
            log.info("Train: Not evaluating model performance on the downstream task, since model is not a VisionLanguageModule.")

        if k_fold_cross_validation:
            # if there is a wandb logger, finish the run, s.t. the next fold gets a new wandb run
            if any(isinstance(l, WandbLogger) for l in logger):
                log.info("Finishing wandb run...")
                wandb.finish()
        else:
            # we return here to skip any more training runs over other folds and the aggregation code below
            return fold_metrics
    
    # here in the code after the loop, we are in k-fold cv mode
    assert k_fold_cross_validation, "This code should only be reached in k-fold cv mode. If this assertion fails, Idk what the fuck happened then"

    aggregated_metrics = defaultdict(list)
    for fold_metrics in all_fold_metrics:
        for k, v in fold_metrics.items():
            aggregated_metrics[k].append(v)

    # Compute mean and std for each metric
    summary_metrics = {
        f"{k}/mean": np.mean(vs) for k, vs in aggregated_metrics.items()
    }
    summary_metrics.update({
        f"{k}/std": np.std(vs) for k, vs in aggregated_metrics.items()
    })

    # log one summary run just for aggregated metrics
    log.info("Train: Instantiating loggers...")
    # if there is a wandb logger, add the group to it
    with open_dict(cfg.logger): # this disables the struct mode of the config (needed in order to set a new key, that was not in the original config)
        if logger_config and isinstance(logger_config, DictConfig):
            for _, lg_conf in logger_config.items():
                if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                    if "WandbLogger" in lg_conf._target_:
                        lg_conf.group = cross_validation_group
                        lg_conf.name = "cross_validation_summary"
                        break  # there shouldnt be more than one wandb logger anyways
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    for lg in logger:
        experiment = lg.experiment
        experiment.log(aggregated_metrics)
        experiment.log(summary_metrics)

    return summary_metrics

def plot_and_wandb_log_train_and_val_tsne_and_confusion_matrix_and_calculate_silhouette_score(model, datamodule, wandb_used, trainer):
    """
    Plot t-SNE visualizations, confusion matrices, and calculate silhouette scores for both training and validation data.
    based on the best model checkpoint.

    Args:
        model: The PyTorch Lightning model instance used for training.
        datamodule: The data module containing training and validation datasets.
        wandb_used (bool): Flag indicating whether Weights & Biases logging is enabled.
        trainer: The PyTorch Lightning trainer instance containing checkpoint callbacks.
    Returns:
        None: This function performs logging and visualization side effects only.
    Note:
        - Prioritizes 'combined' accuracy callback over 'btxrd' accuracy callback for model selection
        - Only executes plotting and logging if both wandb is used and a valid callback is found
        - Logs the following metrics to wandb:
            - t-SNE plots for training and validation data
            - Confusion matrices for training and validation data  
            - Silhouette scores based on tumor labels for training and validation data
            - Silhouette scores based on dataset labels for training and validation data
    """
    # first get the best model based on val/btrxrd/accuracy or if existent on val/combined/accuracy
    val_btxrd_accuracy_callback = None#
    val_combined_accuracy_callback = None
    for cb in trainer.checkpoint_callbacks:
        if 'btxrd' in cb.monitor:
            val_btxrd_accuracy_callback = cb
        if 'combined' in cb.monitor:
            val_combined_accuracy_callback = cb
            break
    
    callback_to_use = None
    if val_btxrd_accuracy_callback is not None:
        callback_to_use = val_btxrd_accuracy_callback
    if val_combined_accuracy_callback is not None: # combined callback has higher priority
        callback_to_use = val_combined_accuracy_callback
    

    # if we have a wandb logger, we can plot the t-SNE of the validation data
    if callback_to_use is not None and wandb_used:
        log.info(f"Train: Plotting t-SNE and confusion matrix and calculating silhouette score of validation data using the version of the model with the best {callback_to_use.monitor}...")
        # get the best model path
        best_model_path = callback_to_use.best_model_path
        # load the model
        best_model = type(model).load_from_checkpoint(best_model_path)

        tsne_figure_val, silhouette_score_based_on_tumor_val, silhouette_score_based_on_dataset_val = plot_validation_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule(best_model, datamodule)
        confusion_matrix_figure_val = plot_validation_confusion_matrix_from_own_model_and_datamodule(best_model, datamodule)
        log.info("Train: Finished plotting t-SNE and confusion matrix and calculating silhouette score of validation data.")
        
        log.info(f"Train: Plotting t-SNE and confusion matrix and calculating silhouette score of train data using the version of the model with the best {callback_to_use.monitor}...")
        tsne_figure_train, silhouette_score_based_on_tumor_train, silhouette_score_based_on_dataset_train = plot_train_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule(best_model, datamodule)
        confusion_matrix_figure_train = plot_train_confusion_matrix_from_own_model_and_datamodule(best_model, datamodule)
        log.info("Train: Finished plotting t-SNE and confusion matrix and calculating silhouette score of train data.")

        wandb.log({
            "tsne_validation": wandb.Image(tsne_figure_val),
            "silhouette_score_based_on_tumor_validation": silhouette_score_based_on_tumor_val,
            "silhouette_score_based_on_dataset_validation": silhouette_score_based_on_dataset_val,
            "confusion_matrix_validation": wandb.Image(confusion_matrix_figure_val),
            "tsne_train": wandb.Image(tsne_figure_train),
            "silhouette_score_based_on_tumor_train": silhouette_score_based_on_tumor_train,
            "silhouette_score_based_on_dataset_train": silhouette_score_based_on_dataset_train,
            "confusion_matrix_train": wandb.Image(confusion_matrix_figure_train)
        })
    else:
        log.info("Train: No t-SNE plots, since no wandb logger was used OR no btxrd callback was found.")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    
    train(cfg)


if __name__ == "__main__":
    main()