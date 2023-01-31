"""
Match transit signals using correlation coefficient tables and additional criteria.
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

    thr_mask = corr_coef_mat_df > match_thr  # matches above matching threshold
    # only match transit signals that only match between them
    match_cnt_mask = np.broadcast_to(np.expand_dims(thr_mask.sum(axis=1), -1), corr_coef_mat_df.shape) + \
                     np.broadcast_to(np.expand_dims(thr_mask.sum(axis=0), 0), corr_coef_mat_df.shape)

    match_mask = (match_cnt_mask == 2) & thr_mask

    idx_row, idx_col = np.where(match_mask)
    for match_idxs_i in range(len(idx_row)):
        matched_signals['signal_a'].append(corr_coef_mat_df.index[idx_row[match_idxs_i]])
        matched_signals['signal_b'].append(corr_coef_mat_df.columns[idx_col[match_idxs_i]])
        matched_signals['match_corr_coef'].append(corr_coef_mat_df.loc[corr_coef_mat_df.index[idx_row[match_idxs_i]],
                                                                       corr_coef_mat_df.columns[idx_col[match_idxs_i]]])

    matched_signals = pd.DataFrame(matched_signals)

    return matched_signals


if __name__ == '__main__':

    match_thr = 0.75
    matching_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/ephemeris_matching_dv/01-27-2023_1132')
    match_dir = matching_root_dir / 'sector_run_tic_tbls'
    matched_signals = []
    for tbl_fp in match_dir.iterdir():

        matched_signals.append(solve_matches(tbl_fp, match_thr))

    matched_signals = pd.concat(matched_signals, axis=0)
    matched_signals.to_csv(matching_root_dir / f'matched_signals_thr{match_thr}.csv', index=False)
