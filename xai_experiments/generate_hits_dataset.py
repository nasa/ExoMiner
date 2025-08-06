# 3rd party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%

preds = pd.read_csv('ranked_predictions_testset.csv')
preds = preds.reset_index(drop=True)
attn_arr = np.load('attn3d.npy', allow_pickle=True)

query_columns = [
    'stellar',
    'dv_tce_fit',
    'global_flux',
    'flux_trend',
    'local_centroid',
    'flux_periodogram',
    'local_odd_even',
    'local_flux',
    'local_weak_secondary',
    'local_unfolded_flux',
    'diff_img',
]

def compute_hits_scores(attn_arr_tce):
    """
    Compute HITS (Hyperlink-Induced Topic Search) scores for a given attention array.
    
    Parameters:
    attn_arr_tce (np.ndarray): Attention array for a specific TCE.
    
    Returns:
    auths (np.ndarray): Authority scores.
    hubs (np.ndarray): Hub scores.
    """
    # Initialize hub and authority scores
    hubs = np.ones(attn_arr_tce.shape[0])
    auths = np.ones(attn_arr_tce.shape[0])

    n_iters_hits = 50
    for hits_iter in range(n_iters_hits):  # or until convergence

        auths_n = attn_arr_tce.T @ hubs
        hubs_n = attn_arr_tce @ auths

        # optional: normalize scores
        auths_n = auths_n / np.linalg.norm(auths_n, ord=2)
        hubs_n = hubs_n / np.linalg.norm(hubs_n, ord=2)

        print(f'Iteration {hits_iter + 1}: {np.linalg.norm(auths - auths_n, ord=2)})')

        auths, hubs = auths_n, hubs_n

    return auths, hubs

all_auths = []
all_hubs = []

for i in range(attn_arr.shape[0]):
    attn_mat = attn_arr[i]  # shape (11, 11) assumed
    auths, hubs = compute_hits_scores(attn_mat)
    all_auths.append(auths)
    all_hubs.append(hubs)

auth_df = pd.DataFrame(all_auths, columns=query_columns)
hub_df = pd.DataFrame(all_hubs, columns=query_columns)

preds_with_auths = pd.concat([preds, auth_df], axis=1)
preds_with_hubs = pd.concat([preds, hub_df], axis=1)

preds_with_auths.to_csv('ranked_predictions_with_authorities.csv', index=False)
preds_with_hubs.to_csv('ranked_predictions_with_hubs.csv', index=False)
