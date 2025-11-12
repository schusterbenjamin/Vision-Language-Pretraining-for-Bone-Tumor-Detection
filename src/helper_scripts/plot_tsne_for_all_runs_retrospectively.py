import os
import glob
import sys
import wandb
from pytorch_lightning import seed_everything

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.baseline.OnlyImagingModule import OnlyImagingModule
from src.data.DownstreamDataModule import DownstreamDataModule
from src.utils.plotting.plot_tsne_and_calculate_silhouette import plot_train_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule, plot_validation_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule
from src.utils.plotting.plot_confusion_matrix import plot_validation_confusion_matrix_from_own_model_and_datamodule, plot_train_confusion_matrix_from_own_model_and_datamodule

ENTITY = "benjamin-schuster"      # e.g. "myusername" or "myteam"
PROJECT = "vision-language-bone-tumor-baseline-imaging"    # e.g. "my_classification_run"

CHECKPOINT_BASE_DIR = "/mnt/nfs/homedirs/benjamins/project/outputs"


def find_checkpoint_for_run(run_id: str) -> str:
    """
    Search CHECKPOINT_BASE_DIR for a file named 'best-checkpoint.ckpt'
    that lives under a subdirectory containing this run_id in its path.

    Returns the full path to best-checkpoint.ckpt if found, else returns ''.
    """
    # We will glob for any path like:
    #   /path/to/lightning_logs/**/<run_id>/**/best-checkpoint.ckpt
    #
    # Adjust the pattern if you used a different naming convention.
    pattern = os.path.join(
        CHECKPOINT_BASE_DIR,
        "**",
        run_id,
        "**",
        "btxrd-*.ckpt"
    )
    matches = glob.glob(pattern, recursive=True)
    if len(matches) == 0:
        return ""
    # If there are multiple matches, choose the first one. You can refine this logic if needed.
    return matches[0]


def main():
    seed_everything(42)

    api = wandb.Api()
    all_runs = api.runs(f"{ENTITY}/{PROJECT}")

    print(f"Found {len(all_runs)} runs under {ENTITY}/{PROJECT}.")

    kfold_module = DownstreamDataModule(
            using_crops=False,
            batch_size=32,
        )
    datamodule, _ = next(kfold_module.get_cv_splits())

    for run in all_runs:
        run_id = run.id
        print(f"\n=== Processing run {run_id} ===")

        # skip if in the configs of the run there is data.try_with_only_n_samples not null
        if run.config.get("data", {}).get("try_with_only_n_samples") is not None:
            print(f"  ➔ Skipping run {run_id} because it has data.try_with_only_n_samples set.")
            continue

        ckpt_path = find_checkpoint_for_run(run_id)
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"  ➔ No checkpoint found for run {run_id} (looked up {ckpt_path}), skipping.")
            continue

        print(f"  ➔ Found checkpoint: {ckpt_path}")

        try:
            model = OnlyImagingModule.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint for run {run_id}: {e}")
            continue

        try:
            tsne_val_fig, silhouette_score_based_on_tumor_val, silhouette_score_based_on_dataset_val = plot_validation_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule(model, datamodule)
            tsne_train_fig, silhouette_score_based_on_tumor_train, silhouette_score_based_on_dataset_train = plot_train_tsne_and_calculate_silhouette_score_from_own_model_and_datamodule(model, datamodule)

            conf_matrix_fig = plot_validation_confusion_matrix_from_own_model_and_datamodule(model, datamodule)
            conf_matrix_train_fig = plot_train_confusion_matrix_from_own_model_and_datamodule(model, datamodule)
        except Exception as e:
            print(f"  ✗ Error while generating t-SNE for run {run_id}: {e}")
            continue

        wandb_run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            id=run_id,
            resume="must",
            reinit=True   # re‐initialize in this process
        )

        # Log the two plots (they come from matplotlib.Figure)
        wandb.log({"tsne_validation": wandb.Image(tsne_val_fig),
            "tsne_train": wandb.Image(tsne_train_fig),
            "silhouette_score_based_on_tumor_validation": silhouette_score_based_on_tumor_val,
            "silhouette_score_based_on_dataset_validation": silhouette_score_based_on_dataset_val,
            "silhouette_score_based_on_tumor_train": silhouette_score_based_on_tumor_train,
            "silhouette_score_based_on_dataset_train": silhouette_score_based_on_dataset_train,
            "confusion_matrix_validation": wandb.Image(conf_matrix_fig),
            "confusion_matrix_train": wandb.Image(conf_matrix_train_fig)})

        # Finish the run so that the next iteration can resume properly
        wandb.finish()
        print(f"  ✔ Logged both t-SNE figures back to run {run_id}.")


if __name__ == "__main__":
    main()
