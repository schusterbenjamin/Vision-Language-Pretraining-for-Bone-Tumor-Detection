import yaml
from pathlib import Path

def latex_escape(s: str) -> str:
    """Escape LaTeX special characters (especially underscores)."""
    if not isinstance(s, str):
        s = str(s)
    return s.replace("_", "\\_")

def load_sweep_params(yaml_path: str):
    """Load sweep parameters and return formatted rows."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    params = config.get("parameters", {})
    rows = []

    for key, val in params.items():
        # Make display names human-friendly
        display_name = key.replace("optimizer.lr", "Learning rate") \
                          .replace("data.batch_size", "Batch size") \
                          .replace("scheduler", "Scheduler") \
                          .replace("optimizer", "Optimizer") \
                          .replace("embedding_dim", "Embedding dimension")
        
        # Determine type and values
        if "values" in val:
            type_ = "Categorical"
            values = ", ".join(latex_escape(str(v)) for v in val["values"])
        elif "min" in val and "max" in val:
            type_ = "Continuous"
            values = f"[{val['min']}, {val['max']}]"
        else:
            type_ = "Unknown"
            values = latex_escape(str(val))

        rows.append((latex_escape(display_name), type_, values))
    return rows


def sweeps_to_latex_table(yaml_paths):
    """Combine multiple sweep configs into one LaTeX table."""
    header = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Hyperparameter & Type & Values / Range \\\\\n"
        "\\midrule\n"
    )

    body = []
    for i, path in enumerate(yaml_paths):
        # Get experiment name from YAML or filename
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        exp_name = config.get("name") or Path(path).stem
        exp_name = latex_escape(exp_name)

        body.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{Experiment: {exp_name}}}}} \\\\")
        rows = load_sweep_params(path)
        for param, type_, values in rows:
            body.append(f"{param} & {type_} & {values} \\\\")
        if i < len(yaml_paths) - 1:
            body.append("\\midrule")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Hyperparameter sweep configurations across multiple experiments.}\n"
        "\\label{tab:multi-sweep-hparams}\n"
        "\\end{table}"
    )

    return header + "\n".join(body) + "\n" + footer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine multiple W&B sweep configs into one LaTeX table."
    )
    parser.add_argument("yaml_paths", nargs="+", help="List of sweep YAML files")
    parser.add_argument("--output", type=str, default=None, help="Optional output .tex file")
    args = parser.parse_args()

    latex_code = sweeps_to_latex_table(args.yaml_paths)
    print(latex_code)

    if args.output:
        Path(args.output).write_text(latex_code)
        print(f"LaTeX table written to {args.output}")
