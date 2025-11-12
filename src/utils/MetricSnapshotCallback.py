import torch
import lightning as pl
from lightning import Callback
import wandb
import logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('project')


class SnapshotAllMetricsOnBestCallback(Callback):
    """
    After each validation epoch, look at `monitor` (e.g. "val/btxrd/accuracy").
    If it is better than all previous epochs, write every metric in
    `trainer.callback_metrics` into wandb.summary with keys:
       "{monitor}_best_{metric_name}"

    The idea is to know all the other metrics as well at the stage where the monitored metric achieved its best value.
    """

    def __init__(self, monitor: str, mode: str):
        """
        Args:
            monitor: the exact key in `self.log(...)` you’re tracking 
                     (e.g. "val/btxrd/accuracy").
            mode:    "max" → higher is better, or "min" → lower is better.
        """
        super().__init__()
        self.monitor = monitor
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.mode = mode
        self.best_val: float = float("-inf") if mode == "max" else float("inf")

    # by having it on validation_end instead of validation_epoch_end, metrics that are logged inside on_validation_epoch_end are also included (in my case thats the combined metrics)
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        This hook is called automatically at the end of validation. At this point,
        `trainer.callback_metrics` contains every metric you logged with `self.log(...)`
        (training + validation) in that epoch. We fetch `monitor` from it, compare
        to self.best_val, and if it improved we dump the entire dict into wandb.summary.
        """
        if trainer.sanity_checking:
            # Skip if we are in the sanity check phase
            return

        metrics = trainer.callback_metrics  # a dict: {metric_name: Tensor or float}

        # 1. Get the current value of the monitored metric
        current = metrics.get(self.monitor)
        if current is None:
            # If your `monitor` key wasn’t logged or is spelled incorrectly, skip.
            return

        # Convert to Python float if needed
        if isinstance(current, torch.Tensor):
            current_val = current.item()
        elif isinstance(current, (float, int)):
            current_val = float(current)
        else:
            # If Lightning returned something unexpected (e.g. a list), bail.
            return

        # 2. Check if “improved” based on mode
        improved = False
        if self.mode == "max" and current_val > self.best_val:
            improved = True
        elif self.mode == "min" and current_val < self.best_val:
            improved = True

        if not improved:
            return
        logger.info(f"SnapshotAllMetricsOnBestCallback: Improved {self.monitor} from {self.best_val} to {current_val}. Snapshotting all metrics to the wandb summary.")

        # 3. Update best_val
        self.best_val = current_val

        # 4. Fetch the wandb.Run object
        try:
            wb_run: wandb.sdk.wandb_run.Run = pl_module.logger.experiment
        except AttributeError:
            logger.warning("SnapshotAllMetricsOnBestCallback: No wandb logger found. Skipping snapshotting.")
            return

        # 5. Loop over ALL metrics in callback_metrics and write them to summary
        for key, val in metrics.items():
            # if Lightning injected any weird keys, skip them
            if key is None:
                continue

            # Convert tensor→float, leave floats/ints alone
            if isinstance(val, torch.Tensor):
                scalar = val.item()
            elif isinstance(val, (float, int)):
                scalar = float(val)
            else:
                # skip anything that isn’t a scalar
                continue

            # Build the new summary key: "val/btxrd/accuracy_best_{metric_name}"
            summary_key = f"{self.monitor}_best_{key}"
            wb_run.summary[summary_key] = scalar

        # No need to call wb_run.summary.update() here; Lightning’s WandbLogger
        # will sync summary back to the server automatically.