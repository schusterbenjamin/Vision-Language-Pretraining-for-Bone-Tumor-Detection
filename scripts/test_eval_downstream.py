import os
import subprocess
import sys
import argparse

import logging
import logging.config

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

logger = logging.getLogger("project")
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.baseline.FusionModule import FusionModule
from src.models.baseline.OnlyImagingModule import OnlyImagingModule
from src.data.DownstreamDataModule import DownstreamDataModule

model_checkpoint_folder = "model_checkpoints/"


def collect_probs(model_checkpoints: list, dataloaders, save_failures: str = None):
    """
    Collect prediction probabilities from the model checkpoints on the test dataloaders.
    Args:
        model_checkpoints (list): model checkpoint file paths.
        dataloaders (list): List of DataLoaders for the test data for each fold.
        save_failures (str): Folder to save falsely predicted x-ray images. If None, do not save.
    Returns:
        dfs (list): List of DataFrames containing the predictions for each fold.
    """
    assert len(model_checkpoints) == len(dataloaders), "Number of model checkpoints must match number of dataloaders"

    already_failed_false_positive=0
    already_failed_false_negative=0
    save_first_n_failures=10

    dfs = []

    for i, (model_checkpoint, dataloader) in enumerate(zip(model_checkpoints, dataloaders)):
        logger.info(f"Test Eval: Evaluating model checkpoint: {i+1}/{len(model_checkpoints)} - {model_checkpoint}")
        # get the model
        try:
            model = OnlyImagingModule.load_from_checkpoint(model_checkpoint)
        except Exception as e:
            try:
                model = FusionModule.load_from_checkpoint(model_checkpoint)
            except Exception as e2:
                logger.error(f"Failed to load model from checkpoint {model_checkpoint}: {e}, {e2}")
                sys.exit(1)

        model.eval()
        model.freeze()

        # evaluate
        result_df = pd.DataFrame(columns=["dataset", "entity", "anatomy_site", "sex", "age", "age_encoded", "tumor", "prob"])
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model = model.cuda()
            if isinstance(model, OnlyImagingModule):
                outputs = model(batch["x-ray"])
            elif isinstance(model, FusionModule):
                outputs, _ = model(batch["x-ray"], batch["age_encoded"], batch["sex_encoded"], batch["anatomy_site_encoded"])
            probs = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
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
                    # add age group from 0-18, 19-39, and 40+
                    "age_group": [age_group],
                    "age_encoded": [batch["age_encoded"][j].item()],
                    "tumor": [batch["tumor"][j].item()],
                    "prob": [probs[j].item()]
                })], ignore_index=True)

                if save_failures:
                    pred = 1 if probs[j].item() >= 0.5 else 0
                    if pred != batch["tumor"][j].item():
                        save = False
                        if pred == 1: # false positive
                            if already_failed_false_positive < save_first_n_failures // 2:
                                save = True
                                already_failed_false_positive += 1
                        else: # false negative
                            if already_failed_false_negative < save_first_n_failures // 2:
                                save = True
                                already_failed_false_negative += 1
                        if save:
                            # save x-ray image with filename including failed number, true label, predicted label, probability and metadata
                            filename = f"failure_{already_failed_false_negative+already_failed_false_positive-1}_true_label_{batch['tumor'][j].item()}_pred_{pred}_prob_{probs[j].item():.4f}_dataset_{batch['dataset'][j]}_entity_{batch['entity'][j]}_anatomy_{batch['anatomy_site'][j]}_age_{batch['age'][j].item()}_sex_{batch['sex'][j]}.png"
                            filepath = os.path.join(save_failures, filename)


                            x_ray_image = batch["x-ray"][j].detach().cpu().permute(1, 2, 0).squeeze().numpy()
                            # normalize to 0-1
                            x_ray_image = (x_ray_image - x_ray_image.min()) / (x_ray_image.max() - x_ray_image.min())
                            plt.imsave(filepath, x_ray_image, cmap='gray')

        dfs.append(result_df)
        logger.info(f"Test Eval: Completed evaluation for model checkpoint: {model_checkpoint}")

    return dfs
            

def evaluate_results(output_file, dfs: list):
    """
    Evaluate the results from the linear probe predictions and compute metrics overall and per subgroup.
    Args:
        output_file (str): Path to save the evaluation results CSV file.
        dfs (list): List of DataFrames containing the predictions for each fold.
    """

    metric_results = pd.DataFrame(columns=["level", "group", "fold", "metric", "value"])

    for fold, df in enumerate(dfs):
        assert "tumor" in df.columns and "prob" in df.columns and "entity" in df.columns and "anatomy_site" in df.columns and "dataset" in df.columns and "sex" in df.columns and "age" in df.columns and "age_encoded" in df.columns,  "DataFrame must contain 'tumor', 'prob', 'entity', 'anatomy_site', 'dataset', 'sex', 'age', and 'age_encoded' columns"


        # overall metrics
        y_true = df["tumor"].values.astype(float)
        y_probs = df["prob"].values

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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    metric_results.to_csv(output_file, index=False, na_rep="NaN")
    logging.info(f"Test Eval: Saved evaluation results to {output_file}")

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
        print(f"Test Eval: Multiple checkpoints found for run_id {run_id}:")
        for i, file in enumerate(checkpoint_files):
            print(f"{i}: {file}")
        choice = int(input("Test Eval: Enter the number of the checkpoint to use: "))
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
    logger.info(f"Test Eval: Saved occurrence counts to {output_file}")

def main(output_file, run_ids: list, torchxrayvision_used: bool = False, save_predictions: str = None, save_failures: str = None):
    """
    Test Evaluation script for evaluating model checkpoints on downstream dataset.

    1. Searches in the outputs/ folder for the given run_ids to find the model checkpoints.
    2. Instantiates the model checkpoints
    3. Instantiates the Downstream dataset.
    4. Evaluates the models on the test set.
    5. Computes metrics overall and per subgroup (dataset, entity, anatomy_site, sex, age_encoded, age_group) and saves a dataframe with the structure (level, group, fold, metric, value).
    6. If specified, saves the raw probabilities without already computed metrics to a specified folder.
    7. If specified, saves the first 10 falsely predicted x-ray images to a specified folder.
    """

    # NOTE: if there are multiple run_ids, I assume that they are from k-fold cv and that they are in the same order as the folds in DownstreamDataModule
    logger.info(f"Test Eval: Evaluating run_ids: {run_ids}")
    model_checkpoints = [get_model_checkpoint(run_id) for run_id in run_ids]
    logger.info(f"Test Eval: Collected model checkpoints: {model_checkpoints}")

    if not torchxrayvision_used:
        data_module = DownstreamDataModule(using_crops=False, batch_size=128)
    else:
        # for torchxrayvision models, we need to scale the intensity normalization to [-1024, 1024] isntead of zero mean and unit std
        data_module = DownstreamDataModule(using_crops=False, batch_size=128, scale_intensity_normalization=True, num_channels=1, try_with_only_n_samples=100)
    test_data_loaders = []
    for fold in range(len(run_ids)):
        test_data_loaders.append(data_module.test_dataloader(fold))
    logger.info(f"Test Eval: Prepared test dataloader with {len(test_data_loaders[0].dataset)} samples")

    if save_failures:
        os.makedirs(save_failures, exist_ok=True)
    result_dfs = collect_probs(model_checkpoints, test_data_loaders, save_failures)
    logger.info(f"Test Eval: Collected probabilities from models")
    if save_predictions:
        os.makedirs(save_predictions, exist_ok=True)
        for i, result_df in enumerate(result_dfs):
            prediction_file = os.path.join(save_predictions, f"predictions_fold_{i}.csv")
            result_df.to_csv(prediction_file, index=False)
        logger.info(f"Test Eval: Saved predictions to {save_predictions}")

    # save occurrences for values of different attribues.
    # take the first result_df, as the occurences are the same for all folds
    # get the same folder as output_file, just use occurences.csv as filename
    # occurences_file = os.path.join(os.path.dirname(output_file), "occurrences.csv")
    # count_and_save_occurences(result_dfs[0], ["dataset", "entity", "anatomy_site", "sex", "age_encoded"], occurences_file)

    evaluate_results(output_file, result_dfs)
    logger.info(f"Test Eval: Completed evaluation of results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Evaluate model checkpoints on downstream dataset. It searches in the outputs/ folder for the given run_ids to find the model checkpoints.
        It is assumed that the run_ids correspond to models trained in the same training just on different folds.
        It instantiates the models and the Downstream dataset. It evaluates the models on the test set.
        It computes metrics overall and per subgroup (dataset, entity, anatomy_site, sex, age_encoded, age_group) and stores it in a dataframe with the structure (level, group, fold, metric, value).
        If you want to have the raw probabilities without already computed metrics, you can specify a folder with the flag --save-predictions.
        You can also save the first 10 falsely predicted x-ray images to a specified folder with the flag --save-failures.
        """
    )

    parser.add_argument("output_file", type=str, help="Path to the output CSV file for evaluation results.")

    parser.add_argument("--torchxrayvision", action=argparse.BooleanOptionalAction, help="Whether the models were trained using torchxrayvision")
    parser.set_defaults(torchxrayvision=False)

    parser.add_argument("--save-predictions", type=str, help="Folder to save individual predictions CSV files in. If not provided, predictions will not be saved.")
    parser.set_defaults(save_predictions=None)

    parser.add_argument("--save-failures", type=str, help="Save the first 10 falsely predicted x-ray images to the specified folder.")
    parser.set_defaults(save_failures=None)

    parser.add_argument("run_ids", nargs='+', type=str, help="List of run IDs to evaluate")

    args = parser.parse_args()

    output_file = args.output_file
    torchxrayvision_used = args.torchxrayvision
    save_predictions = args.save_predictions
    save_failures = args.save_failures
    run_ids = args.run_ids

    main(output_file, run_ids, torchxrayvision_used, save_predictions, save_failures)