"""
Match transit signals using correlation coefficient tables.

A match is accepted if 1) the correlation score is higher than the defined threshold and 2) it is the highest score out
of all matches done for a given transit signal.

Each match table should be such that the first column is the ID of the transit signals that are to be matched to a
single transit signal from the set of transit signals defined in the columns after the ID column of the row transit
signal. Each match table should only contain transit signals that are related to the same target star.

e.g.

|         uid            | col_transit_signal_1 | col_transit_signal_2  |
| row_transit_signal_1   |     corr_row1_col1   |     corr_row1_col2    |
| row_transit_signal_2   |     corr_row2_col1   |    corr_row2_col2     |

where corr_rowX_colY refers to the correlation score between transit signal in row X and transit signal in row Y.
Match is accepted between these two if:

-> corr_rowX_colY > matching_threshold AND max(corr_rowX_coli) = corr_rowX_colY
### (deprecated) -> corr_rowX_colY > matching_threshold AND all corr_rowX_coli <= matching_threshold, for i != Y

"""

# 3rd party
import numpy as np
import pandas as pd
from pathlib import Path


def solve_matches(tbl_fp, match_thr):
    """ Solve matches among groups of transit signals.

    Args:
        tbl_fp: Path, filepath to match table
        match_thr: float, threshold used for accepting a match

    Returns: matched_signals, pandas DataFrame with accepted matches

    """

    matched_signals = {'signal_a': [], 'signal_b': [], 'match_corr_coef': []}

    corr_coef_mat_df = pd.read_csv(tbl_fp, index_col=0)  # load correlation table
    corr_coef_mat_df = corr_coef_mat_df.fillna(-1)  # set NaNs to -1

    # thr_mask = corr_coef_mat_df > match_thr  # matches above matching threshold
    # # only match transit signals that only match between them
    # match_cnt_mask = np.broadcast_to(np.expand_dims(thr_mask.sum(axis=1), -1), corr_coef_mat_df.shape) + \
    #                  np.broadcast_to(np.expand_dims(thr_mask.sum(axis=0), 0), corr_coef_mat_df.shape)
    #
    # # matches signals whose matching score between them is above the threshold and it is the only one above it for both
    # # of them
    # match_mask = (match_cnt_mask == 2) & thr_mask
    #
    # idx_row, idx_col = np.where(match_mask)
    # for match_idxs_i in range(len(idx_row)):
    #     matched_signals['signal_a'].append(corr_coef_mat_df.index[idx_row[match_idxs_i]])
    #     matched_signals['signal_b'].append(corr_coef_mat_df.columns[idx_col[match_idxs_i]])
    #     matched_signals['match_corr_coef'].append(corr_coef_mat_df.loc[corr_coef_mat_df.index[idx_row[match_idxs_i]],
    #                                                                    corr_coef_mat_df.columns[idx_col[match_idxs_i]]])

    idxs_cols_maxcorr = np.argmax(corr_coef_mat_df.to_numpy(), axis=1)
    for row_i, idx_col_maxcorr in enumerate(idxs_cols_maxcorr):
        if corr_coef_mat_df.loc[corr_coef_mat_df.index[row_i], corr_coef_mat_df.columns[idx_col_maxcorr]] > match_thr:
            matched_signals['signal_a'].append(corr_coef_mat_df.index[row_i])
            matched_signals['signal_b'].append(corr_coef_mat_df.columns[idx_col_maxcorr])
            matched_signals['match_corr_coef'].append(
                corr_coef_mat_df.loc[corr_coef_mat_df.index[row_i], corr_coef_mat_df.columns[idx_col_maxcorr]])

        check_for_multiple_matches = corr_coef_mat_df.loc[corr_coef_mat_df.index[row_i]] > match_thr
        multiple_matches = check_for_multiple_matches[check_for_multiple_matches].index.to_list()
        n_multiple_matches = len(multiple_matches)
        if n_multiple_matches > 1:
            print(f'Signal {corr_coef_mat_df.index[row_i]} has a correlation score > {match_thr} '
                  f'for {n_multiple_matches} signals: {multiple_matches}.')

    matched_signals = pd.DataFrame(matched_signals)

    return matched_signals


if __name__ == '__main__':

    match_thr = 0.75  # set matching threshold
    # get file paths to match tables for multiple sector runs
    matching_root_dir = Path('')
    match_dir = matching_root_dir / 'sector_run_tic_tbls'
    matched_signals = []
    for tbl_fp in match_dir.iterdir():  # iterate through sector run match tables.

        matched_signals.append(solve_matches(tbl_fp, match_thr))

    # aggregate match results into a single file
    matched_signals = pd.concat(matched_signals, axis=0)
    matched_signals.to_csv(matching_root_dir / f'matched_signals_thr{match_thr}.csv', index=False)
