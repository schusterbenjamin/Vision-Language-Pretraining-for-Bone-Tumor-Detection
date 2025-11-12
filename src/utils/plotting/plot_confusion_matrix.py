import os
import sys
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.models.baseline.FusionModule import FusionModule
from src.models.baseline.OnlyImagingModule import OnlyImagingModule
from src.data.DownstreamDataModule import DownstreamDataModule
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix_from_model_and_data(model, dataloaders, device, dataset_name):
    """
    Generates and plots a normalized confusion matrix for a given model and dataloaders.
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloaders (list of torch.utils.data.DataLoader): List of dataloaders containing the data to evaluate.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        dataset_name (str): Name of the dataset, used for the plot title.
    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the confusion matrix heatmap.
    Notes:
        - Assumes each batch is a dictionary with keys 'x-ray' (input tensor) and 'tumor' (ground truth labels).
        - The function applies a sigmoid activation and a threshold of 0.5 to obtain binary predictions.
        - The confusion matrix is normalized and displayed as percentages.
    """
    model.eval()

    with torch.no_grad():
        predictions = []
        labels = []
        for dataloader in dataloaders:
            for batch in dataloader:
                x_ray_tensors = batch['x-ray'].to(device)
                age_encoded, sex_encoded, anatomy_site_encoded = batch['age_encoded'].to(device), batch['sex_encoded'].to(device), batch['anatomy_site_encoded'].to(device)
                tumor = batch['tumor']

                if isinstance(model, OnlyImagingModule):
                    logits = model(x_ray_tensors)
                elif isinstance(model, FusionModule):
                    logits, _ = model(x_ray_tensors, age_encoded, sex_encoded, anatomy_site_encoded)
                else:
                    raise ValueError("Model type not recognized. Expected OnlyImagingModule or FusionModule.")

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                predictions.extend(preds.tolist())
                labels.extend(tumor.tolist())

        cf_matrix = confusion_matrix(labels, predictions)
        
        fig = plt.figure(figsize=(10, 8))
        plt.title(f"{dataset_name}: Confusion Matrix")
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        return fig
    
def plot_validation_confusion_matrix_from_own_model_and_datamodule(model, datamodule):
    device = model.device
    val_dataloaders = datamodule.val_dataloader()
    fig = plot_confusion_matrix_from_model_and_data(model, val_dataloaders, device, 'Validation')
    return fig

def plot_train_confusion_matrix_from_own_model_and_datamodule(model, datamodule):
    device = model.device
    train_dataloaders = [datamodule.train_dataloader()]
    fig = plot_confusion_matrix_from_model_and_data(model, train_dataloaders, device, 'Train')
    return fig

def plot_confusion_matrix_from_checkpoint(ckpt_path, datamodule):
    model = OnlyImagingModule.load_from_checkpoint(ckpt_path)
    device = model.device
    val_dataloaders = datamodule.val_dataloader()
    fig = plot_confusion_matrix_from_model_and_data(model, val_dataloaders, device, 'Validation')
    return fig

def plot_confusion_matrix_from_checkpoint(ckpt_path):
    model = OnlyImagingModule.load_from_checkpoint(ckpt_path)
    device = model.device

    kfold_module = DownstreamDataModule(using_crops=False, batch_size=16)

    datamodule, _ = next(kfold_module.get_cv_splits())
    val_dataloaders = datamodule.val_dataloader()

    fig = plot_confusion_matrix_from_model_and_data(model, val_dataloaders, device, 'Validation')
    return fig
    
if __name__ == "__main__":
    ckpt_path = "/home/benjamins/project/outputs/2025-05-17/14-33-32/vision-language-bone-tumor-baseline-imaging/uw3j3mp2/checkpoints/btxrd-epoch:61-val_internal_loss:0.185-val_btxrd_loss:0.63.ckpt"
    model: OnlyImagingModule = OnlyImagingModule.load_from_checkpoint(ckpt_path)
    network = model.network
    device = model.device

    kfold_module = DownstreamDataModule(using_crops=False, batch_size=2, try_with_only_n_samples=16)

    datamodule, _ = next(kfold_module.get_cv_splits())
    val_dataloaders = datamodule.val_dataloader()

    fig = plot_confusion_matrix_from_model_and_data(network, val_dataloaders, device, 'Validation')
    fig.savefig("confusion_matrix_plot.png")


        