import os
import sys
from sklearn.metrics import silhouette_score
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.models.baseline.OnlyImagingModule import OnlyImagingModule
from src.data.DownstreamDataModule import DownstreamDataModule
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tsne_and_calculate_silhouette_score_from_model_and_data(model, dataloaders, device, dataset_name):
    """
    Extracts features from a model using provided dataloaders, computes t-SNE embeddings, calculates silhouette scores, and plots the results.
    Args:
        model: The neural network model with a `forward_features` method for feature extraction.
        dataloaders (list): List of PyTorch DataLoader objects providing batches with 'x-ray', 'dataset', and 'tumor' keys.
        device: The device (e.g., 'cuda' or 'cpu') to perform computations on.
        dataset_name (str): Name of the dataset, used for plot titling.
    Returns:
        fig (matplotlib.figure.Figure): The generated t-SNE scatter plot figure.
        sil_score_based_on_tumor (float): Silhouette score calculated using tumor labels.
        sil_score_based_on_dataset (float): Silhouette score calculated using dataset labels.
    Notes:
        - Features are extracted from the model and pooled if necessary.
        - t-SNE is used to reduce feature dimensionality to 2D for visualization.
        - The plot colors points by dataset and uses marker style for tumor labels.
        - Silhouette scores provide a measure of clustering quality based on tumor and dataset labels.
    """
    
    with torch.no_grad():
        features = None
        dataset_info = []
        tumor_info = []
        for dataloader in dataloaders:
            for batch in dataloader:
                x_ray_tensors = batch['x-ray'].to(device)
                dataset = batch['dataset']
                tumor = batch['tumor']

                extracted_features = model.forward_features(x_ray_tensors)
                # if features have more than 2 dimensions, average pool the dimensions that are "too much"
                if len(extracted_features.shape) > 2:
                    extracted_features = torch.mean(extracted_features, dim=(2, 3))

                if features is None:
                    features = extracted_features
                else:
                    features = torch.cat([features, extracted_features], dim=0)
                dataset_info.extend(dataset)
                tumor_info.extend(tumor)

        features_np = features.cpu().numpy()
        tumor_labels = [t.item() for t in tumor_info]

        sil_score_based_on_tumor = silhouette_score(features_np, tumor_labels)
        sil_score_based_on_dataset = silhouette_score(features_np, dataset_info)

        if len(features_np) < 30:
            perplexity = len(features_np) - 1
        else:
            perplexity = 30
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_results = tsne.fit_transform(features_np)

        # Prepare DataFrame
        df = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2'])
        df['dataset'] = dataset_info
        df['tumor'] = tumor_labels

        # Plot
        fig = plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df,
            x='tsne-1',
            y='tsne-2',
            hue='dataset',
            style='tumor',
            palette='tab10',
            s=60
        )
        plt.title(f"{dataset_name}: t-SNE of features colored by Dataset (color) and Tumor (marker)")
        plt.text(
            0.99, 0.01,
            f'Silhouette Score Based on Tumor/No Tumor: {sil_score_based_on_tumor:.4f}\nSilhouette Score Based on Dataset: {sil_score_based_on_dataset:.4f}',
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig, sil_score_based_on_tumor, sil_score_based_on_dataset
    

def plot_validation_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule(model, datamodule):
    network = model.get_image_network()
    device = model.device
    val_dataloaders = datamodule.val_dataloader()
    return plot_tsne_and_calculate_silhouette_score_from_model_and_data(network, val_dataloaders, device, 'Validation')

def plot_train_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule(model, datamodule):
    network = model.get_image_network()
    device = model.device
    train_dataloaders = [datamodule.train_dataloader()]
    return plot_tsne_and_calculate_silhouette_score_from_model_and_data(network, train_dataloaders, device, 'Train')

def plot_tsne_and_calculate_silhouette_score_from_checkpoint(ckpt_path, datamodule):
    model = OnlyImagingModule.load_from_checkpoint(ckpt_path)
    network = model.get_image_network()
    device = model.device
    val_dataloaders = datamodule.val_dataloader()
    return plot_tsne_and_calculate_silhouette_score_from_model_and_data(network, val_dataloaders, device, 'Validation')

def plot_tsne_and_calculate_silhouette_score_from_checkpoint(ckpt_path):
    model = OnlyImagingModule.load_from_checkpoint(ckpt_path)
    network = model.get_image_network()
    device = model.device

    kfold_module = DownstreamDataModule(using_crops=False, batch_size=16)

    datamodule, _ = next(kfold_module.get_cv_splits())
    val_dataloaders = datamodule.val_dataloader()

    return plot_tsne_and_calculate_silhouette_score_from_model_and_data(network, val_dataloaders, device, 'Validation')


if __name__ == "__main__":
    ckpt_path = "/home/benjamins/project/outputs/2025-05-17/14-33-32/vision-language-bone-tumor-baseline-imaging/uw3j3mp2/checkpoints/btxrd-epoch:61-val_internal_loss:0.185-val_btxrd_loss:0.63.ckpt"
    model: OnlyImagingModule = OnlyImagingModule.load_from_checkpoint(ckpt_path)
    network = model.network
    device = model.device

    kfold_module = DownstreamDataModule(using_crops=False, batch_size=2, try_with_only_n_samples=16)

    datamodule, _ = next(kfold_module.get_cv_splits())
    val_dataloaders = datamodule.val_dataloader()

    fig, silhouett_score = plot_tsne_and_calculate_silhouette_score_from_model_and_data(network, val_dataloaders, device, 'Validation')
    fig.savefig("tsne_plot.png")

