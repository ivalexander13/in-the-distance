from src.plot_stressor_regimes import plot_stressor_regimes
plot_stressor_regimes(
    random_seeds = range(5),
    stressors={
        "numchars": [10],
        "numstates": [5],
        "mut_prop": [0.1],
        "drop_total": [0],
    },
)