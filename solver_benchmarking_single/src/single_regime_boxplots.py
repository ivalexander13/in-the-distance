import os
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def collect_metrics(stubs2names: List[Tuple[str, str]], out_folder: str = "./data/") -> pd.DataFrame:
    """Collect metrics from the output directory.

    Args:
        stubs2names (List[Tuple[str, str]]): the stubs to names mapping.
        out_folder (str, optional): output directory. Defaults to "./data/".

    Returns:
        pd.DataFrame: the metrics dataframe.
    """
    metrics_df = pd.DataFrame()
    for name, _ in stubs2names:
        metrics_df[f"RF__{name}"] = pd.read_csv(
            os.path.join(out_folder, f"{name}.rf.csv")
        )["NormalizedRobinsonFoulds"]
        metrics_df[f"triplets__{name}"] = pd.read_csv(
            os.path.join(out_folder, f"{name}.triplets.csv")
        ).groupby(['Replicate']).mean()["TripletsCorrect"]

    return metrics_df

def single_regime_boxplots(
    stubs2names: List[Tuple[str, str]],
    out_folder: str = "./data/",
    plots_folder: str = "./plots/",
    plot_file_suffix: str = "benchmark",
) -> Tuple[str, str]:
    """Plot boxplots for the single regime benchmark.

    Args:
        stubs2names (List[Tuple[str, str]]): the stubs to names mapping.
        out_folder (str, optional): output directory. Defaults to "./data/".
        plots_folder (str, optional): plot images directory. Defaults to "./plots/".
        plot_file_suffix (str, optional): suffix for plot image file. Defaults to "benchmark".

    Returns:
        Tuple[str, str]: the plot image file paths.
    """
    # Collect Metrics
    metrics_df = collect_metrics(stubs2names, out_folder)

    # Plot metrics
    rf_vars = {
        "prefix": "RF",
        "title": "Benchmark: Normalized Robinson Foulds Distance",
        "x_label": "Normalized RF Distance (Smaller is Better)",
    }
    triplets_vars = {
        "prefix": "triplets",
        "title": "Benchmark: Triplets Correct",
        "x_label": "Triplets Correct (Smaller is Better)",
    }

    # Reverse list to preserve original order in plot
    stubs2names = stubs2names.copy()
    stubs2names.reverse()

    # Plotting
    plot_files = []
    for vars in [rf_vars, triplets_vars]:
        names = []
        yticks = []
        for name, ytick in stubs2names:
            names.append(metrics_df[f'{vars["prefix"]}__{name}'])
            yticks.append(ytick)

        # Get plot file
        plot_file = os.path.join(plots_folder, f'{vars["prefix"]}.{plot_file_suffix}.png')
        plot_files.append(plot_file)


        plt.clf()
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.boxplot(names, vert=0)
        plt.title(vars["title"])
        plt.xlabel(vars["x_label"])
        plt.yticks(list(range(1, len(metrics_df.columns) // 2 + 1)), yticks)
        plt.xlim(-0.05, 1.05)
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.show()

    return tuple(plot_files)
