from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.benchmark import get_stressor_by_params, get_score
from src.parameters import TreeParameters, LineageTracingParameters, SolverParameters


def _collect_scores_from_files(algs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine scores from all algorithms into one dataframe.

    Args:
        algs (List[str]): List of solver names 

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Combined dataframes.
    """
    # Get and append all score dataframes.
    SCORES_DIR = Path("solver_benchmarking_whole_regime/data/scores/")

    RF_df = pd.DataFrame()
    triplets_df = pd.DataFrame()
    for alg in algs:
        temp_RF = pd.read_csv(SCORES_DIR / f"{alg}.robinson_foulds.tsv", sep='\t')
        RF_df = pd.concat([RF_df, temp_RF])
        RF_df['Score'] = RF_df['NormalizedRobinsonFoulds']

        temp_triplets = pd.read_csv(SCORES_DIR / f"{alg}.triplets_correct.tsv", sep='\t')
        triplets_df = pd.concat([triplets_df, temp_triplets])
        triplets_df['Score'] = triplets_df['TripletsCorrect']

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
        "Stressor",
        "Parameter"
    ]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """Transform dataframes into a format that is easier to plot.

    Args:
        RF_df (pd.DataFrame): Dataframe with Robinson-Foulds scores.
        triplets_df (pd.DataFrame): Dataframe with triplets scores.
        group_cols_by (list, optional): Columns with which the dataframes should be grouped by. Defaults to [ "Algorithm", "Replicate", "Fitness", "Stressor", "Parameter" ].
        name_col_includes (list, optional): the columns merged to form the name column. Defaults to [ "Stressor", "Parameter" ].

    Returns:
        RF_df: Transformed dataframe with Robinson-Foulds scores.
        triplets_df: Transformed dataframe with triplets scores.
        name_col_includes: List of columns that will be merged into the "name" column (y axis).
    """

    RF_df_grouped = RF_df.groupby(group_cols_by, as_index=False).mean()
    triplets_df_grouped = triplets_df.groupby(group_cols_by, as_index=False).mean()

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
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()

    return outfile


def plot_stressor_regimes(
    topology_type='exponential_plus_c',
    numcells = 2000,
    fitness_regimes = ['no_fit', 'high_fit'],

    stressors = {
        "numchars": [10, 20, 60, 90, 150],
        "numstates": [5, 10, 25, 50, 500],
        "mut_prop": [0.1, 0.3, 0.7, 0.9],
        "drop_total": [0, 0.1, 0.3, 0.4, 0.5, 0.6],
    },

    numchars__default = 40,
    numstates__default = 100,
    mut_prop__default = 0.5,
    drop_total__default = 0.2,

    random_seeds = range(50),
    
    solver_names = ['nj', 'nj_iwhd', 'nj_iwhd_oracle'], # order matters

    solver_plot_params: Dict[str, Dict[str, Any]] = {
        "nj_iwhd_oracle": {
            "box_color": "green",
        },
        'nj_iwhd': {
            "box_color": "firebrick",
        },
        'nj': {
            "box_color": "royalblue",
        }
    },

    plot_params: Dict[str, Any] = {
        "name_col_includes": ["Stressor", "Parameter"], # To unsplit, add 'Fitness' to this list
        "split_plots_by": 'Fitness', # To unsplit, set to None
        "figsize": (10, 12),
        "hue": 'Algorithm',
        "width": 0.7,
        "linewidth": 0.7,
        "plot_dir": "./plots/",
    }

):
    """Constructs boxplots for the performance of solvers on datasets of various stressors.

    Args:
        topology_type (str, optional): Defaults to 'exponential_plus_c'.
        numcells (int, optional): Number of cells, usually 400 or 2000. Defaults to 2000.
        fitness_regimes (list, optional): Fitness regimes, including no_fit, low_fit, and high_fit.
        stressors (dict, optional): Variations in tree parameters, for each one having other params held constant. Defaults are the ones that does not require running the simulator.
        numchars__default (int, optional): Default value to use when not a stressor. Defaults to 40.
        numstates__default (int, optional): Default value to use when not a stressor. Defaults to 100.
        mut_prop__default (float, optional): Default value to use when not a stressor. Defaults to 0.5.
        drop_total__default (float, optional): Default value to use when not a stressor. Defaults to 0.2.
        random_seeds (optional): Tree number. Defaults to range(50).
        solver_names (list, optional): Solver names. Defaults to ['nj', 'nj_iwhd', 'nj_iwhd_oracle'].
        plot_params (_type_, optional): Parameters for plotting. Defaults to { "name_col_includes": ["Stressor", "Parameter"], # To unsplit, add 'Fitness' to this list "split_plots_by": 'Fitness', # To unsplit, set to None "figsize": (10, 12), "hue": 'Algorithm', "width": 0.7, "linewidth": 0.7, "plot_dir": "./plots/", }.

    Returns:
        outfiles: the filepaths of the plots generated by this function
    """
    data_cols = [
        "Fitness",
        "Stressor",
        "Parameter",
        "Algorithm",
        "Replicate",
        "Score"
    ]

    #################### Uncomment to Run Cascade #############################
    rf_data = []
    triplets_data = []

    total_iterations = len(fitness_regimes) * len(random_seeds) * len(solver_names) * sum([len(val) for val in stressors.values()])
    pbar = tqdm(total=total_iterations)
    for nfitness_regime, \
        nseed, \
        nsolver_name \
        in product(
            fitness_regimes,
            random_seeds,
            solver_names):

        for stressor, list_values in stressors.items():
            nchars = numchars__default
            nstates = numstates__default
            mut_prop = mut_prop__default
            drop_total = drop_total__default

            for param in list_values:
                pbar.set_description(f"{nsolver_name} > {nfitness_regime} > tree{nseed} > {stressor}{param}")

                if stressor == 'numchars':
                    nchars = param
                elif stressor == 'numstates':
                    nstates = param
                elif stressor == 'mut_prop':
                    mut_prop = param
                elif stressor == 'drop_total':
                    drop_total = param

                # Construct Parameters 
                tree_parameters = TreeParameters(
                    topology_type=topology_type,
                    n_cells=numcells,
                    fitness=nfitness_regime,
                    random_seed=nseed,
                )

                lt_parameters = LineageTracingParameters(
                    numchars=nchars,
                    numstates=nstates,
                    drop_total=drop_total,
                    mut_prop=mut_prop,
                    random_seed=nseed,
                )

                solver_parameters = SolverParameters(
                    solver_name=nsolver_name,
                    collapse_mutationless_edges=True,
                    priors_type='no_priors'
                )

                # Get score
                stressor, param = get_stressor_by_params(lt_parameters)
                rf_data.append(
                    [
                        nfitness_regime,
                        stressor,
                        param,
                        nsolver_name,
                        nseed,
                        get_score(
                            'rf',
                            tree_parameters = tree_parameters,
                            lt_parameters = lt_parameters,
                            solver_parameters = solver_parameters,
                        )
                    ]
                )
                triplets_data.append(
                    [
                        nfitness_regime,
                        stressor,
                        param,
                        nsolver_name,
                        nseed,
                        get_score(
                            'triplets',
                            tree_parameters = tree_parameters,
                            lt_parameters = lt_parameters,
                            solver_parameters = solver_parameters,
                        )
                    ]
                )

                pbar.update(1)

    rf_df = pd.DataFrame(rf_data, columns=data_cols)
    triplets_df = pd.DataFrame(triplets_data, columns=data_cols)

    #################### Uncomment to Use Cached Scores #############################
    # rf_df, triplets_df = _collect_scores_from_files(solver_names)
    # rf_df = rf_df[data_cols]
    # triplets_df = triplets_df[data_cols]
    ####################################################################

    rf_df, triplets_df, name_col_includes = _transform_df(
        rf_df, 
        triplets_df, 
        name_col_includes=plot_params['name_col_includes']
    )

    # Plot
    # Get aesthetic params for plotting
    palette = {alg: solver_plot_params[alg]['box_color'] for alg in solver_names}
    split_plots_by = plot_params['split_plots_by']

    # Plot Configs
    RF_vars = {
        'df': rf_df,
        'title': 'Benchmark: Normalized Robinson Foulds Distance',
        'x_label': 'Normalized RF Distance (Smaller is Better)',
        'out_prefix': 'rf_benchmark',
    }
    triplets_vars = {
        'df': triplets_df,
        'title': 'Benchmark: Triplets Correct',
        'x_label': 'Triplets Correct (Larger is Better)',
        'out_prefix': 'triplets_benchmark',
    }

    # Do plotting
    outfiles = []
    for vars in [RF_vars, triplets_vars]:
        if split_plots_by is None:
            outfile = _plot(
                    df=vars['df'],
                    colname='Score',
                    title=vars['title'],
                    x_label=vars['x_label'],
                    outfile=plot_params['plot_dir'] + vars['out_prefix'] + '.png',
                    palette=palette,
                    hue_order=solver_names,
                    name_col_includes=name_col_includes,
                    figsize=plot_params['figsize'],
                    hue=plot_params['hue'],
                    width=plot_params['width'],
                    linewidth=plot_params['linewidth']
                )

            outfiles.append(outfile)
        else:
            for split in vars['df'][split_plots_by].unique():
                outfile = _plot(
                    df=vars['df'][vars['df'][split_plots_by] == split],
                    colname='Score',
                    title=vars['title'] + f' ({split})',
                    x_label=vars['x_label'],
                    outfile=plot_params['plot_dir'] + vars['out_prefix'] + f'.{split}.png',
                    palette=palette,
                    hue_order=solver_names,
                    name_col_includes=name_col_includes,
                    figsize=plot_params['figsize'],
                    hue=plot_params['hue'],
                    width=plot_params['width'],
                    linewidth=plot_params['linewidth']
                )

                outfiles.append(outfile)

    return outfiles

                



         