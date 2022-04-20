from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def _collect_scores(algs: List[Dict[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine scores from all algorithms into one dataframe.

    Args:
        algs (List[Dict[str, str]]): List of dictionaries with algorithm names and scores.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Combined dataframes.
    """
    # Get and append all score dataframes.
    RF_df = pd.DataFrame()
    triplets_df = pd.DataFrame()
    for alg in algs:
        temp_RF = pd.read_csv(Path(alg['scores_dir']) / f"{alg['alg_name']}.robinson_foulds.tsv", sep='\t')
        RF_df = pd.concat([RF_df, temp_RF])

        temp_triplets = pd.read_csv(Path(alg['scores_dir']) / f"{alg['alg_name']}.triplets_correct.tsv", sep='\t')
        triplets_df = pd.concat([triplets_df, temp_triplets])

    return (RF_df, triplets_df)

def _transform_df(
    RF_df: pd.DataFrame, 
    triplets_df: pd.DataFrame,
    group_cols_by = [
        "Algorithm",
        "Replicate",
        "Fitness",
        "Stressor",
        "Parameter"
    ],
    name_col_includes = [
        # "Fitness",
        "Stressor",
        "Parameter"
    ]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """Transform dataframes into a format that is easier to plot.

    Args:
        RF_df (pd.DataFrame): Dataframe with Robinson-Foulds scores.
        triplets_df (pd.DataFrame): Dataframe with triplets scores.
        group_cols_by (list, optional): Columns with which the dataframes should be grouped by. Defaults to [ "Algorithm", "Replicate", "Fitness", "Stressor", "Parameter" ].
        name_col_includes (list, optional): _description_. Defaults to [ "Stressor", "Parameter" ].

    Returns:
        RF_df: Transformed dataframe with Robinson-Foulds scores.
        triplets_df: Transformed dataframe with triplets scores.
        name_col_includes: List of columns that will be merged into the "name" column (y axis).
    """

    RF_df_grouped = RF_df.groupby(group_cols_by, as_index=False).mean()
    triplets_df_grouped = triplets_df.groupby(group_cols_by, as_index=False).mean().drop(columns=['Depth'])

    def apply_name(row):
        list_name = [str(row[col]) for col in name_col_includes]
        return ' | '.join(list_name)

    RF_df_grouped['name'] = RF_df_grouped.apply(apply_name, axis=1)
    triplets_df_grouped['name'] = triplets_df_grouped.apply(apply_name, axis=1)

    return (RF_df_grouped, triplets_df_grouped, name_col_includes)

def _get_stressor_dividers(
    df: pd.DataFrame, 
    colname: str, 
    repeats: int, 
    name_col_includes: List[str]
    ) -> List[float]:
    """Get section dividers for a given stressor, for its plot.

    Args:
        df (pd.DataFrame): Dataframe with scores.
        colname (str): Name of the stressor column to divide by.
        repeats (int): Number of duplicates the stressor will be showed in one plot. If the stressor shows up for both high_fit and no_fit Fitness regimes, then repeats=2 if the plot includes both fitness regimes, whereas repeats=1 if the plot only includes one fitness regime.
        name_col_includes (List[str]): List of columns that will be merged into the "name" column (y axis). This is to figure out the number of groups in the plot.

    Returns:
        List[float]: Y coordinates for the divider lines.
    """
    out = df.groupby(name_col_includes, as_index=False).count()[colname].value_counts().sort_index().values
    out = np.tile(out, repeats)
    out = out[:-1].cumsum()
    out = out / repeats - 0.5
    return out

def _plot(
    df: pd.DataFrame,
    colname: str, 
    title: str,
    x_label: str,
    outfile: str,
    palette: Dict[str, str],
    hue_order: List[str],
    name_col_includes: List[str],
    figsize=(10, 12),
    hue='Algorithm',
    width=0.7,
    linewidth=0.7
    ) -> str:
    """Plot scores for a given dataframe (single score and split, if applicable).

    Args:
        df (pd.DataFrame): Dataframe with scores.
        colname (str): Name of the score column to plot.
        title (str): Title of the plot.
        x_label (str): X axis label.
        outfile (str): Path to the output plot file.
        palette (Dict[str, str]): Dictionary with algorithm names as keys and colors as values.
        hue_order (List[str]): Order of the algorithms in the plot.
        name_col_includes (List[str]): List of columns that will be merged into the "name" column (y axis).
        figsize (tuple, optional): Figure size. Defaults to (10, 12).
        hue (str, optional): Which column to make the boxes be side by side. Defaults to 'Algorithm'.
        width (float, optional): Width of boxes. Defaults to 0.7.
        linewidth (float, optional): Width of outlines. Defaults to 0.7.

    Returns:
        str: Path to the output plot file.
    """
    sns.set(rc={'figure.figsize': figsize})

    # Main Boxplot
    bp = sns.boxplot(
        y='name', 
        x=colname,
        data=df, 
        hue=hue, 
        width=width,
        palette=palette,
        hue_order=hue_order,
        linewidth=linewidth
        )

    # Plot Dividers
    stressor_divisions = _get_stressor_dividers(df, 'Stressor', 1, name_col_includes)
    fitness_divisions = _get_stressor_dividers(df, 'Fitness', 1, name_col_includes) 
    [plt.axhline(y, color = 'r', linestyle='--') for y in stressor_divisions]
    [plt.axhline(y, color = 'b', linestyle='--') for y in fitness_divisions]

    # Visuals
    plt.title(title)
    plt.xlabel(x_label)
    plt.savefig(outfile, dpi=300)
    plt.show()

    return outfile

def plot_scores(
    algs: List[Dict[str, str]],
    name_col_includes=['Fitness', 'Stressor', 'Parameter'],
    split_plots_by='Fitness',
    figsize=(10, 12),
    hue='Algorithm',
    width=0.7,
    linewidth=0.7
    ) -> List[str]:
    """Plot scores for a given list of algorithms.

    Args:
        algs (List[Dict[str, str]]): List of dictionaries with algorithm names (alg_name), the box color (box_color), and the parent directory to its score file (score_dir).
        name_col_includes (list, optional): Columns to merge into the "name" column. Defaults to ['Fitness', 'Stressor', 'Parameter'].
        split_plots_by (str, optional): Create multiple plots by splitting on this column. Defaults to 'Fitness'.
        figsize (tuple, optional): Figure size. Defaults to (10, 12).
        hue (str, optional): Which column to make the boxes be side by side. Defaults to 'Algorithm'.
        width (float, optional): Width of boxes. Defaults to 0.7.
        linewidth (float, optional): Thickness of outlines. Defaults to 0.7.

    Returns:
        List[str]: Paths to the output plots.
    """
    RF_df, triplets_df = _collect_scores(algs)
    RF_df, triplets_df, name_col_includes = _transform_df(RF_df, triplets_df, name_col_includes=name_col_includes)

    # Get aesthetic params for plotting
    palette = {alg['alg_name']: alg['box_color'] for alg in algs}
    hue_order = [alg['alg_name'] for alg in algs]

    # Plot Configs
    RF_vars = {
        'df': RF_df,
        'title': 'Benchmark: Normalized Robinson Foulds Distance',
        'x_label': 'Normalized RF Distance (Smaller is Better)',
        'outfile': 'plots/rf_benchmark',
        'colname': 'NormalizedRobinsonFoulds'
    }
    triplets_vars = {
        'df': triplets_df,
        'title': 'Benchmark: Triplets Correct',
        'x_label': 'Triplets Correct (Larger is Better)',
        'outfile': 'plots/triplets_benchmark',
        'colname': 'TripletsCorrect'
    }

    # Do plotting
    outfiles = []
    for vars in [RF_vars, triplets_vars]:
        if split_plots_by is None:
            outfile = _plot(
                    df=vars['df'],
                    colname=vars['colname'],
                    title=vars['title'],
                    x_label=vars['x_label'],
                    outfile=vars['outfile'] + '.png',
                    palette=palette,
                    hue_order=hue_order,
                    name_col_includes=name_col_includes,
                    figsize=figsize,
                    hue=hue,
                    width=width,
                    linewidth=linewidth
                )

            outfiles.append(outfile)
        else:
            for split in vars['df'][split_plots_by].unique():
                outfile = _plot(
                    df=vars['df'][vars['df'][split_plots_by] == split],
                    colname=vars['colname'],
                    title=vars['title'] + f' ({split})',
                    x_label=vars['x_label'],
                    outfile=vars['outfile'] + f'.{split}.png',
                    palette=palette,
                    hue_order=hue_order,
                    name_col_includes=name_col_includes,
                    figsize=figsize,
                    hue=hue,
                    width=width,
                    linewidth=linewidth
                )

                outfiles.append(outfile)

    return outfiles