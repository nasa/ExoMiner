""" Utility functions for analyzing results from AUM experiments. """

# 3rd party
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def overlap(r1, r2, d_i):
    """ Compute overlap between rankings `r1` and `r2` at depth `d`, i.e., the number of common elements that show up in
    both rankings up to depth `d`.

    :param r1: NumPy array, ranking 1
    :param r2: NumPy array, ranking 2
    :param d_i: int, depth
    :return:
        float, overlap
    """
    return len(np.intersect1d(r1[:d_i], r2[:d_i])) / d_i


# def rbo(r1, r2, d, p):
#
#     return (1 - p) * np.sum([p ** (d_i - 1) * overlap(r1, r2, d_i) for d_i in range(1, d + 1)])


def avg_overlap(r1, r2, d):
    """ Compute average overlap between rankings `r1` and `r2` at a cut-off depth `d`.

    :param r1: NumPy array, ranking 1
    :param r2: NumPy array, ranking 2
    :param d: int, cut-off depth
    :return:
        float, average overlap
    """

    return 1 / d * np.sum([overlap(r1, r2, d_i) for d_i in range(1, d + 1)])


def count_on_top_k(example, top_k):
    """ Count number of times example is in top-k across runs.

    :param example: pandas Series, rankings for example across runs
    :param top_k: int, top-k
    :return:
        int, number of times examples is in top-k across runs.
    """

    return (example < top_k).sum()


def plot_logits_margins_aum_example_all_runs(example, n_runs, n_epochs, experiment_dir, examples_dir):
    """ Plot logits, margins and AUM values over epochs for all runs for each example.

    :param example: tuple (kic, tce_plnt_num), TCE ID
    :param n_runs: int, number of runs
    :param n_epochs: int, number of training epochs
    :param experiment_dir: Path, experiment directory
    :param examples_dir: Path, directory to save results
    :return:
    """

    plt.ioff()
    print(f'Plotting for example {example["uid"]}...')

    aum_tbl_allruns = pd.read_csv(experiment_dir / f'aum_allruns_epoch{n_epochs - 1}.csv')
    aum_tbl_allruns.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)
    logits = [0, 1]
    score_og_label = []
    epoch_arr = np.arange(n_epochs)
    f, ax = plt.subplots(3, 1, figsize=(8, 8))

    logits_dict, margins_arr, aum_arr = {logit_i: np.nan * np.ones((n_runs, n_epochs)) for logit_i in logits}, \
                                        np.nan * np.ones((n_runs, n_epochs)),  \
                                        np.nan * np.ones((n_runs, n_epochs))
    for run in range(n_runs):

        print(f'[{example["uid"]}] Run {run}')

        run_dir = experiment_dir / 'runs' / f'run{run}'
        model_dir = run_dir / 'models' / 'model1'

        # get AUM over epochs
        aum_tbl = pd.read_csv(run_dir / 'models' / 'model1' / 'aum.csv')
        aum_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

        ranking_tbl = pd.read_csv(model_dir / 'ranking_alldatasets.csv')
        ranking_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

        # grab model score for this example
        score_og_label.append(ranking_tbl.loc[example["uid"], f'score_{aum_tbl.loc[example["uid"], "original_label"]}'])

        # get logits over epochs tables
        logits_tbls = {logit_i: pd.read_csv(model_dir / f'logit{logit_i}_allepochs.csv') for logit_i in logits}
        for logit_i in logits_tbls:
            logits_tbls[logit_i] = logits_tbls[logit_i].set_index(keys=['target_id', 'tce_plnt_num'])

        # get margins over epochs table
        margins_dir = model_dir / 'margins'
        margins_tbl = pd.read_csv(margins_dir / 'margins_allepochs.csv')
        margins_tbl.set_index(keys=['target_id', 'tce_plnt_num'], inplace=True)

        # plot logits
        for logit_i in logits_tbls:
            logits_dict[logit_i][run] = logits_tbls[logit_i].loc[example["uid"], [f'epoch_{epoch_i}' for epoch_i in epoch_arr]]
            ax[0].plot(epoch_arr,
                       logits_dict[logit_i][run],
                       label=f'Logit {logit_i}' if run == 0 else None,
                       color='orange' if logit_i == 1 else 'b', alpha=0.3)
        # plot margins
        margins_arr[run] = margins_tbl.loc[example["uid"], [f'epoch_{epoch_i}' for epoch_i in epoch_arr]]
        ax[1].plot(epoch_arr,
                   margins_arr[run], color='b', alpha=0.3)
        # plot AUM values
        aum_arr[run] = aum_tbl.loc[example["uid"], [f'epoch_{epoch_i}' for epoch_i in epoch_arr]]
        ax[2].plot(epoch_arr,
                   aum_arr[run], color='b', alpha=0.3)

    for logit_i in logits:
        ax[0].plot(epoch_arr, np.mean(logits_dict[logit_i], axis=0), color='orange' if logit_i == 1 else 'b', linestyle='--')
    ax[1].plot(epoch_arr, np.mean(margins_arr, axis=0), color='r', linestyle='--')
    ax[2].plot(epoch_arr, np.mean(aum_arr, axis=0), color='r', linestyle='--')
    ax[0].set_xlim([0, n_epochs])
    ax[1].set_xlim([0, n_epochs])
    ax[2].set_xlim([0, n_epochs])
    ax[2].set_xlabel('Epoch Number')
    ax[0].set_ylabel('Logit Value')
    ax[1].set_ylabel('Margin')
    ax[2].set_ylabel('AUM')
    ax[0].set_title(f'Example {example["uid"]} {aum_tbl_allruns.loc[example["uid"], "original_label"]}\n'
                    f'{aum_tbl_allruns.loc[example["uid"], "dataset"]} dataset | '
                    f'Score {np.mean(score_og_label):.3f}+-{np.std(score_og_label, ddof=1):.3f} | '
                    # f'Predicted class {ranking_tbl.loc[example, "predicted class"]}\n'
                    f'{example["label"]}\n '
                    f'AUM: {aum_tbl_allruns.loc[example["uid"], "mean"]:.3f}+-{aum_tbl_allruns.loc[example["uid"], "std"]:.3f}')
    ax[0].legend()
    f.savefig(examples_dir /
              f'{example["uid"]}-{aum_tbl_allruns.loc[example["uid"], "original_label"]}_logits_margin_aum.png')
    # plt.close()
