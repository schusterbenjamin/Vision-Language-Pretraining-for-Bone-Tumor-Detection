import os
import subprocess
import sys
import argparse

import logging
import logging.config

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import lightning as L
from torch.utils.data import DataLoader

logger = logging.getLogger("project")
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.pretrain.VisionLanguageModule import VisionLanguageModule
from src.data.DownstreamDataModule import DownstreamDataModule
from src.utils.LinearProbeCallback import LinearProbeCallback

model_checkpoint_folder = "model_checkpoints/"


def collect_probs(image_encoder, classification_heads, dataloaders):
    """
    Collect prediction probabilities from the linear probe classification heads on the test dataloaders.
    Args:
        image_encoder: The image encoder model.
        classification_heads (list): List of trained LogisticRegression classifiers for each fold.
        dataloaders (list): List of DataLoaders for the test data for each fold.
    Returns:
        dfs (list): List of DataFrames containing the predictions for each fold."""
    assert len(classification_heads) == len(dataloaders), "Number of classification heads must match number of dataloaders"

    dfs = []

    for i, (classification_head, dataloader) in enumerate(zip(classification_heads, dataloaders)):
        logger.info(f"Linear Probe Linear Probe Test Eval: Evaluating linear probe classification head: {i+1}/{len(classification_heads)}")
        image_encoder.eval()

        # evaluate
        result_df = pd.DataFrame(columns=["dataset", "entity", "anatomy_site", "sex", "age", "age_encoded", "tumor", "prob"])
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                image_encoder = image_encoder.cuda()
            
            features = image_encoder(batch["x-ray"])  # [B, D]
            # Keep features on CPU for sklearn
            probs = classification_head.predict_proba(features.cpu().numpy())[:, 1]  # [B,] - already probabilities, no sigmoid needed


            for j in range(len(batch["dataset"])):
                if batch['age'][j].item() < 19:
                        age_group = '0-18'
                elif 19 <= batch['age'][j].item() < 40:
                    age_group = '19-39'
                else:
                    age_group = '40+'
                    
                result_df = pd.concat([result_df, pd.DataFrame({
                    "dataset": [batch["dataset"][j]],
                    "entity": [batch["entity"][j]],
                    "anatomy_site": [batch["anatomy_site"][j]],
                    "sex": [batch["sex"][j]],
                    "age": [batch["age"][j].item()],
                    "age_encoded": [batch["age_encoded"][j].item()],
                    "age_group": [age_group],
                    "tumor": [batch["tumor"][j].item()],
                    "prob": [probs[j].item()]
                })], ignore_index=True)

        dfs.append(result_df)
        logger.info(f"Linear Probe Test Eval: Completed evaluation for classification head {i+1}/{len(classification_heads)}")

    return dfs
            

def evaluate_results(output_file, dfs: list):
    """
    Evaluate the results from the linear probe predictions and compute metrics overall and per subgroup.
    Args:
        output_file (str): Path to save the evaluation results CSV file.
        dfs (list): List of DataFrames containing the predictions for each fold.
    """
    metric_results = pd.DataFrame(columns=["level", "fold", "metric", "value"])

    for fold, df in enumerate(dfs):
        assert "tumor" in df.columns and "prob" in df.columns and "entity" in df.columns and "anatomy_site" in df.columns and "dataset" in df.columns and "sex" in df.columns and "age" in df.columns and "age_encoded" in df.columns,  "DataFrame must contain 'tumor', 'prob', 'entity', 'anatomy_site', 'dataset', 'sex', 'age', and 'age_encoded' columns"


        # overall metrics
        y_true = df["tumor"].values.astype(float)
        y_probs = df["prob"].values
        print(y_true, y_probs)

        overall_metrics = calculate_metrics(y_true, y_probs)
        for metric, value in overall_metrics.items():
            metric_results = pd.concat([metric_results, pd.DataFrame({
                "level": ["overall"],
                "group": ["overall"],
                "fold": [fold],
                "metric": [metric],
                "value": [value]
            })], ignore_index=True)
        
        # per dataset metrics
        for dataset in df["dataset"].unique():
            df_subset = df[df["dataset"] == dataset]
            y_true_subset = df_subset["tumor"].values.astype(float)
            y_probs_subset = df_subset["prob"].values
            dataset_metrics = calculate_metrics(y_true_subset, y_probs_subset)
            for metric, value in dataset_metrics.items():
                metric_results = pd.concat([metric_results, pd.DataFrame({
                    "level": ["dataset"],
                    "group": [dataset],
                    "fold": [fold],
                    "metric": [metric],
                    "value": [value]
                })], ignore_index=True)

        # per entity metrics
        for entity in df["entity"].unique():
            df_subset = df[df["entity"] == entity]
            y_true_subset = df_subset["tumor"].values.astype(float)
            y_probs_subset = df_subset["prob"].values
            entity_metrics = calculate_metrics(y_true_subset, y_probs_subset)
            for metric, value in entity_metrics.items():
                metric_results = pd.concat([metric_results, pd.DataFrame({
                    "level": ["entity"],
                    "group": [entity],
                    "fold": [fold],
                    "metric": [metric],
                    "value": [value]
                })], ignore_index=True)
        
        # per anatomy_site metrics
        for anatomy_site in df["anatomy_site"].unique():
            df_subset = df[df["anatomy_site"] == anatomy_site]
            y_true_subset = df_subset["tumor"].values.astype(float)
            y_probs_subset = df_subset["prob"].values
            anatomy_site_metrics = calculate_metrics(y_true_subset, y_probs_subset)
            for metric, value in anatomy_site_metrics.items():
                metric_results = pd.concat([metric_results, pd.DataFrame({
                    "level": ["anatomy_site"],
                    "group": [anatomy_site],
                    "fold": [fold],
                    "metric": [metric],
                    "value": [value]
                })], ignore_index=True)

        # per sex metrics
        for sex in df["sex"].unique():
            df_subset = df[df["sex"] == sex]
            y_true_subset = df_subset["tumor"].values.astype(float)
            y_probs_subset = df_subset["prob"].values
            sex_metrics = calculate_metrics(y_true_subset, y_probs_subset)
            for metric, value in sex_metrics.items():
                metric_results = pd.concat([metric_results, pd.DataFrame({
                    "level": ["sex"],
                    "group": [sex],
                    "fold": [fold],
                    "metric": [metric],
                    "value": [value]
                })], ignore_index=True)

        # per age group metrics (age_encoded)
        for age_encoded in df["age_encoded"].unique():
            df_subset = df[df["age_encoded"] == age_encoded]
            y_true_subset = df_subset["tumor"].values.astype(float)
            y_probs_subset = df_subset["prob"].values
            age_encoded_metrics = calculate_metrics(y_true_subset, y_probs_subset)
            for metric, value in age_encoded_metrics.items():
                metric_results = pd.concat([metric_results, pd.DataFrame({
                    "level": ["age_encoded"],
                    "group": [age_encoded],
                    "fold": [fold],
                    "metric": [metric],
                    "value": [value]
                })], ignore_index=True)

        # per age group metrics (age_group)
        for age_group in df["age_group"].unique():
            df_subset = df[df["age_group"] == age_group]
            y_true_subset = df_subset["tumor"].values.astype(float)
            y_probs_subset = df_subset["prob"].values
            age_group_metrics = calculate_metrics(y_true_subset, y_probs_subset)
            for metric, value in age_group_metrics.items():
                metric_results = pd.concat([metric_results, pd.DataFrame({
                    "level": ["age_group"],
                    "group": [age_group],
                    "fold": [fold],
                    "metric": [metric],
                    "value": [value]
                })], ignore_index=True)

    # save metric_results to csv
    # create parent directory if it does not exist
    # but only if output_file is not just a filename in the current directory
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    metric_results.to_csv(output_file, index=False, na_rep="NaN")
    logging.info(f"Linear Probe Test Eval: Saved evaluation results to {output_file}")

def calculate_metrics(y_true, y_probs):
    """
    Calculate evaluation metrics given true labels and predicted probabilities.
    Args:
        y_true (np.ndarray): True binary labels.
        y_probs (np.ndarray): Predicted probabilities for the positive class.
    Returns:
        dict: Dictionary containing evaluation metrics."""

    y_pred = (y_probs >= 0.5).astype(int)

    metrics = {}
    
    auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics["accuracy"] = acc
    metrics["balanced_accuracy"] = balanced_acc

    if len(set(y_true)) < 2:
        metrics["roc_auc"] = float('nan')
        metrics["precision"] = float('nan')
        metrics["recall"] = float('nan')
        metrics["f1_score"] = float('nan')
    else:
        metrics["roc_auc"] = auc
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1

    return metrics
    

def find_folder_with_run_id(base_folder: str, run_id: str) -> str:
    """
    Find folders within the base_folder that match the given run_id using the Unix find command.
    Args:
        base_folder (str): The base directory to search within.
        run_id (str): The run ID to search for.
    Returns:
        list: List of folder paths that match the run_id.
    """
    try:
        # Execute the find command
        result = subprocess.run(
            ['find', base_folder, '-type', 'd', '-name', run_id],
            capture_output=True,
            text=True,
            check=True
        )
        # Return the output
        return result.stdout.strip().split('\n') if result.stdout else []
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []

def get_model_checkpoint(run_id: str) -> str:
    """
    Get the model checkpoint file path for the given run_id.
    If multiple checkpoints are found, prompt the user to select one.
    Args:
        run_id (str): The run ID of the model.
    Returns:
        str: Path to the selected model checkpoint file."""
    run_folder = find_folder_with_run_id(model_checkpoint_folder, run_id)
    assert len(run_folder) == 1, f"Expected one folder for run_id {run_id}, found {len(run_folder)}"

    # get the files in run_folder/checkpoints
    checkpoints_folder = os.path.join(run_folder[0], "checkpoints")
    checkpoint_files = os.listdir(checkpoints_folder)

    # if there are multiple checkpoints, ask user to pick one
    if len(checkpoint_files) > 1:
        print(f"Linear Probe Test Eval: Multiple checkpoints found for run_id {run_id}:")
        for i, file in enumerate(checkpoint_files):
            print(f"{i}: {file}")
        choice = int(input("Linear Probe Test Eval: Enter the number of the checkpoint to use: "))
        assert 0 <= choice < len(checkpoint_files), "Invalid choice"
        checkpoint_file = checkpoint_files[choice]
    else:
        checkpoint_file = checkpoint_files[0]

    return os.path.join(checkpoints_folder, checkpoint_file)
    

def count_and_save_occurences(df: pd.DataFrame, identifiers: list, output_file: str):
    """
    Count occurrences of each value for the given identifiers in the dataframe and save to CSV.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        identifiers (list): List of column names to count occurrences for.
        output_file (str): Path to save the occurrences CSV file.
    """
    for identifier in identifiers:
        assert identifier in df.columns, f"Identifier {identifier} not in dataframe columns"

    counts_df = pd.DataFrame(columns=["attribute", "value", "count"])
    for identifier in identifiers:
        value_counts = df[identifier].value_counts()
        for value, count in value_counts.items():
            counts_df = pd.concat([counts_df, pd.DataFrame({
                "attribute": [identifier],
                "value": [value],
                "count": [count]
            })], ignore_index=True)
    
    sum_counts = []
    for identifier in identifiers:
        total_count = counts_df[counts_df["attribute"] == identifier]["count"].sum()
        sum_counts.append(total_count)
    assert len(set(sum_counts)) == 1, "Sum of counts for each identifier should be the same"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    counts_df.to_csv(output_file, index=False)
    logger.info(f"Linear Probe Test Eval: Saved occurrence counts to {output_file}")

def train_linear_probes_k_fold(image_encoder, data_module: DownstreamDataModule, device):
    """
    Train linear probes on each fold of the downstream dataset using k-fold cross-validation.
    Args:
        image_encoder: The image encoder model.
        data_module: The DownstreamDataModule providing the data.
        device: Device to perform computations on.
    Returns:
        (classification_heads, val_accs, val_aurocs) (list, list, list): List of trained LogisticRegression classifiers for each fold, list of validation accuracies, list of validation AUROCs.
    """

    classification_heads = []
    val_accs, val_aurocs = [], []

    for data_module_folds, _ in data_module.get_cv_splits():
        train_dataloader = data_module_folds.train_dataloader()
        val_datasets = [dl.dataset for dl in data_module_folds.val_dataloader()]
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        val_dataloader = DataLoader(val_dataset, batch_size=train_dataloader.batch_size, shuffle=False)

        classification_head, val_acc, val_auroc = _linear_probe_training(image_encoder, train_dataloader, val_dataloader, device)
        classification_heads.append(classification_head)
        val_accs.append(val_acc)
        val_aurocs.append(val_auroc)

    return classification_heads, val_accs, val_aurocs
        

def _linear_probe_training(image_encoder, train_dataloader, val_dataloader, device):
        """
        Train a linear probe on extracted features from the image encoder.
        Args:
            image_encoder: The image encoder model.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            device: Device to perform computations on.
        Returns:
            (clf, val_acc, val_auroc): Trained LogisticRegression classifier, Validation accuracy, Validation AUROC
        """
        image_encoder = image_encoder.eval()

        logger.debug(f"LinearProbeTestEval: Extracting features for linear probe training")
        X_train, y_train = _extract_features(image_encoder, train_dataloader, device)
        X_val, y_val = _extract_features(image_encoder, val_dataloader, device)
        logger.debug("LinearProbeTestEval: Extracted features for linear probe training")

        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
        )
        logger.debug(f"LinearProbeTestEval: Training linear probe using LogisticRegression")
        logger.debug(f"LinearProbeTestEval: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)

        y_scores = clf.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, y_scores)

        return clf, val_acc, val_auroc
        

def _extract_features(encoder, dataloader, device):
    """
    Extract features from the encoder for all samples in the dataloader.
    Args:
        encoder: The image encoder model.
        dataloader: DataLoader providing the data.
        device: Device to perform computations on.
    Returns:
        (X, y) (np.ndarray, np.ndarray): Numpy array of extracted features, Numpy array of corresponding labels.
    """
    all_feats, all_labels = [], []
    encoder.eval()  # Ensure eval mode
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features for linear probe"):
            imgs, labels = batch["x-ray"], batch["tumor"]
            imgs = imgs.to(device)
            feats = encoder(imgs)           # [B, D]
            # Immediately move to CPU and convert to numpy
            feats = feats.detach().cpu().numpy()
            all_feats.append(feats)
            all_labels.append(labels.cpu().numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def main(output_file, run_id: str, save_predictions: str = None):
    """
    Linear Probe Test Evaluation Script.

    1. Loads a pre-trained VisionLanguageModule from a specified checkpoint.
    2. Prepares the DownstreamDataModule for evaluation.
    3. Trains linear probes on each fold of the downstream dataset.
    4. Evaluates the trained linear probes on the test set.
    5. Computes and saves evaluation metrics overall and per subgroup (dataset, entity, anatomy_site, sex, age_encoded, age_group).
    6. Optionally saves raw prediction probabilities to specified folder.

    Args:
        output_file (str): Path to save the evaluation results CSV file.
        run_id (str): Run ID of the model to evaluate.
        save_predictions (str, optional): Folder to save individual predictions CSV files in. If not provided, predictions will not be saved.

    """
    L.seed_everything(42)
    logger.info(f"Linear Probe Test Eval: Evaluating run_id: {run_id}")
    model_checkpoint = get_model_checkpoint(run_id)
    logger.info(f"Linear Probe Test Eval: Collected model checkpoints: {model_checkpoint}")

    data_module = DownstreamDataModule(using_crops=False, batch_size=128)
    test_data_loaders = [data_module.test_dataloader(fold=i) for i in range(4)] # we have 4 folds in the downstream dataset
    logger.info(f"Linear Probe Test Eval: Prepared test dataloader with {len(test_data_loaders[0].dataset)} samples")

    vlp_module = VisionLanguageModule.load_from_checkpoint(model_checkpoint)
    vlp_module.eval()
    vlp_module.freeze()


    classification_heads, val_accs, val_aurocs = train_linear_probes_k_fold(vlp_module.image_encoder, data_module, vlp_module.device)
    # the validation performance is just an addition, since the k-fold results have not been computed during training.
    logger.info(f"Linear Probe Test Eval: Trained linear probes with validation accuracies: {val_accs} and AUROCs: {val_aurocs}")
    logger.info(f"Linear Probe Test Eval: Mean validation accuracy: {np.mean(val_accs)} +- {np.std(val_accs)}")
    logger.info(f"Linear Probe Test Eval: Mean validation AUROC: {np.mean(val_aurocs)} +- {np.std(val_aurocs)}")

    result_dfs = collect_probs(vlp_module.image_encoder, classification_heads, test_data_loaders)
    logger.info(f"Linear Probe Test Eval: Collected probabilities from models")
    if save_predictions:
        os.makedirs(save_predictions, exist_ok=True)
        for i, result_df in enumerate(result_dfs):
            prediction_file = os.path.join(save_predictions, f"predictions_fold_{i}.csv")
            result_df.to_csv(prediction_file, index=False)
        logger.info(f"Linear Probe Test Eval: Saved predictions to {save_predictions}")

    # save occurrences for values of different attribues.
    # take the first result_df, as the occurences are the same for all folds
    # get the same folder as output_file, just use occurences.csv as filename
    # occurences_file = os.path.join(os.path.dirname(output_file), "occurrences.csv")
    # count_and_save_occurences(result_dfs[0], ["dataset", "entity", "anatomy_site", "sex", "age_encoded"], occurences_file)

    evaluate_results(output_file, result_dfs)
    logger.info(f"Linear Probe Test Eval: Completed evaluation of results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Evaluate linear probe on downstream dataset. It searches in the outputs/ folder for the given run_id to find the model checkpoint.
        It instantiates the model and the Downstream dataset. It trains linear probes on each fold of the downstream dataset and evaluates them on the test set.
        It computes metrics overall and per subgroup (dataset, entity, anatomy_site, sex, age_encoded, age_group) and stores it in a dataframe with the structure (level, group, fold, metric, value).
        If you want to have the raw probabilities without already computed metrics, you can specify a folder with the flag --save-predictions.
        """
    )
    parser.add_argument("output_file", type=str, help="Path to save the evaluation results CSV file.")
    parser.add_argument("run_id", type=str, help="Run ID of the model to evaluate.")
    parser.add_argument("--save-predictions", type=str, help="Folder to save individual predictions CSV files in. If not provided, predictions will not be saved.")
    parser.set_defaults(save_predictions=None)
    args = parser.parse_args()

    output_file = args.output_file
    run_id = args.run_id
    save_predictions = args.save_predictions

    main(output_file, run_id, save_predictions)