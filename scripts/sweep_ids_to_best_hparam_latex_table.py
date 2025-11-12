import wandb
from pathlib import Path

def latex_escape(s: str) -> str:
    """Escape LaTeX special characters (especially underscores)."""
    if not isinstance(s, str):
        s = str(s)
    return s.replace("_", "\\_")

def natural_name(key: str) -> str:
    """Map sweep keys to natural readable names."""
    return {
        "optimizer.lr": "Learning rate",
        "data.batch_size": "Batch size",
        "scheduler": "Scheduler",
        "optimizer": "Optimizer",
        "model.embedding_dim": "Shared Dimension"
    }.get(key, key)

def pretty_value(key: str, value) -> str:
    """Map hyperparameter values to human-readable strings."""
    if key == "scheduler":
        return {
            "no_scheduler": "No Scheduler",
            "cosine": "Cosine",
            "cosine_with_warmup": "Cosine with Warmup"
        }.get(value, str(value))
    elif key == "optimizer":
        return {
            "adam": "Adam",
            "adamw": "AdamW"
        }.get(value, str(value))
    return str(value)

def get_best_run_from_sweep(sweep_id: str, metric: str):
    """Fetch the best run from a W&B sweep by metric (maximize)."""
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs = sweep.runs

    if not runs:
        print(f"⚠️ No runs found for sweep {sweep_id}")
        return None, None, None

    best_run, best_val = None, float("-inf")
    for run in runs:
        value = run.summary.get(metric)
        if value is not None and value > best_val:
            best_val = value
            best_run = run

    if not best_run:
        print(f"⚠️ No valid metric values found for sweep {sweep_id}")
        return None, None, None

    sweep_params = sweep.config.get("parameters", {}).keys()
    hparams = {k: best_run.config.get(k) for k in sweep_params if k in best_run.config}

    return best_run, best_val, hparams

def best_hparams_to_latex(sweep_ids, metric):
    """Create LaTeX table listing best sweep hyperparameters per sweep in fixed order."""
    header = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "Hyperparameter & Best Value \\\\\n"
        "\\midrule\n"
    )

    # Define fixed row order
    row_order = ["optimizer", "scheduler", "optimizer.lr", "data.batch_size", "model.embedding_dim"]

    body = []
    api = wandb.Api()

    for i, sweep_id in enumerate(sweep_ids):
        sweep = api.sweep(sweep_id)
        sweep_name = sweep.config.get("name") if sweep.config else Path(sweep_id).name
        sweep_name = latex_escape(sweep_name or sweep_id)

        best_run, best_val, hparams = get_best_run_from_sweep(sweep_id, metric)
        if not best_run:
            continue

        # Experiment row with best metric in brackets
        body.append(f"\\multicolumn{{2}}{{l}}{{\\textbf{{Experiment: {sweep_name} [{best_val:.4f}]}}}} \\\\")

        for key in row_order:
            if key in hparams:
                display_name = latex_escape(natural_name(key))
                value = latex_escape(pretty_value(key, hparams[key]))
                body.append(f"{display_name} & {value} \\\\")

        if i < len(sweep_ids) - 1:
            body.append("\\midrule")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Best hyperparameter configurations per sweep based on validation accuracy.}\n"
        "\\label{tab:best-hparams}\n"
        "\\end{table}"
    )

    return header + "\n".join(body) + "\n" + footer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LaTeX table of best hparams per W&B sweep.")
    parser.add_argument("sweep_ids", nargs="+", help="List of sweep IDs (e.g. 'entity/project/sweeps/abc123')")
    parser.add_argument(
        "--metric",
        type=str,
        default="val/combined/accuracy_best_val/combined/accuracy",
        help="Metric name to maximize (default: val/combined/accuracy_best_val/combined/accuracy)",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional .tex output file")

    args = parser.parse_args()

    latex_code = best_hparams_to_latex(args.sweep_ids, args.metric)
    print(latex_code)

    if args.output:
        Path(args.output).write_text(latex_code)
        print(f"LaTeX table written to {args.output}")
