# 3rd party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.metrics import AUC, Precision, Recall, FalsePositives


def compute_prcurve_roc_auc(study_dir, dataset, num_thresholds, basename, plot=False, verbose=False):
    """ Compute PR curve and ROC (also AUC).

    :param study_dir: Path, study directory
    :param dataset: str, dataset
    :param num_thresholds: int, number of thresholds to be used between 0 and 1
    :param basename: str, name to be added to the figure name
    :param plot: bool, if True plots figures
    :param verbose: bool, if True prints data into std out.
    :return:
    """

    save_dir = study_dir / basename
    save_dir.mkdir(exist_ok=True)

    dataset_tbl = pd.read_csv(study_dir / f'ensemble_ranked_predictions_{dataset}set.csv')

    threshold_range = list(np.linspace(0, 1, num=num_thresholds, endpoint=False))

    # compute metrics
    auc_pr = AUC(num_thresholds=num_thresholds,
                 summation_method='interpolation',
                 curve='PR',
                 name='auc_pr')
    auc_roc = AUC(num_thresholds=num_thresholds,
                  summation_method='interpolation',
                  curve='ROC',
                  name='auc_roc')

    _ = auc_pr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    auc_pr = auc_pr.result().numpy()
    _ = auc_roc.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    auc_roc = auc_roc.result().numpy()

    # compute precision and recall thresholds for the PR curve
    precision_thr = Precision(thresholds=threshold_range, top_k=None, name='prec_thr')
    recall_thr = Recall(thresholds=threshold_range, top_k=None, name='rec_thr')
    _ = precision_thr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    precision_thr_arr = precision_thr.result().numpy()
    _ = recall_thr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    recall_thr_arr = recall_thr.result().numpy()

    # compute FPR for the ROC
    false_pos_thr = FalsePositives(thresholds=threshold_range, name='prec_thr')
    _ = false_pos_thr.update_state(dataset_tbl['label'].tolist(), dataset_tbl['score'].tolist())
    false_pos_thr_arr = false_pos_thr.result().numpy()
    fpr_thr_arr = false_pos_thr_arr / len(dataset_tbl.loc[dataset_tbl['label'] == 0])

    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'prec_thr': precision_thr_arr,
        'rec_thr': recall_thr_arr,
        'fpr_thr': fpr_thr_arr
    }

    if verbose:
        print(f'Dataset {dataset}: {metrics}')

    np.save(save_dir / f'metrics_{dataset}.npy', metrics)

    if plot:
        # plot PR curve
        f, ax = plt.subplots()
        ax.plot(recall_thr_arr, precision_thr_arr)
        # ax.scatter(recall_thr_arr, precision_thr_arr, c='r')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.8, 0.1, f'AUC={auc_pr:.3}', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'precision-recall_curve_{dataset}.svg')
        plt.close()

        # plot PR curve zoomed in
        f, ax = plt.subplots()
        ax.plot(recall_thr_arr, precision_thr_arr)
        # ax.scatter(recall_thr_arr, precision_thr_arr, c='r')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 21))
        ax.set_yticks(np.linspace(0, 1, 21))
        ax.set_xlim([0.5, 1])
        ax.set_ylim([0.5, 1])
        ax.text(0.8, 0.6, f'AUC={auc_pr:.3}', bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'precision-recall_curve_zoom_{dataset}.svg')
        plt.close()

        # plot ROC
        f, ax = plt.subplots()
        ax.plot(fpr_thr_arr, recall_thr_arr)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.8, 0.1, 'AUC={:.3}'.format(auc_roc), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'roc_{dataset}.svg')
        plt.close()

        # plot ROC zoomed in
        f, ax = plt.subplots()
        ax.plot(fpr_thr_arr, recall_thr_arr)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 1, 21))
        ax.set_yticks(np.linspace(0, 1, 21))
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0.5, 1])
        ax.text(0.3, 0.6, 'AUC={:.3}'.format(auc_roc), bbox={'facecolor': 'gray', 'alpha': 0.2, 'pad': 10})
        f.savefig(save_dir / f'roc_zoom_{dataset}.svg')
        plt.close()


def compute_precision_at_k(study, dataset, rootDir, k_arr, k_curve_arr, k_curve_arr_plot, basename, plot=False,
                           verbose=False):
    """ Compute precision-at-k and plot related curves.

    :param study: str, study name
    :param dataset: str, dataset. Either 'train', 'val' and 'test'
    :param rootDir: Path, root directory with the studies
    :param k_arr: list with k values for which to compute precision-at-k
    :param k_curve_arr: list with k values for which to compute precision-at-k curve
    :param k_curve_arr_plot: list with values for which to draw xticks (k values)
    :param plot: bool, if True plots precision-at-k and misclassified-at-k curves
    :param verbose: bool, if True print precision-at-k values
    :return:
    """

    save_dir = rootDir / study / basename
    save_dir.mkdir(exist_ok=True)

    rankingTbl = pd.read_csv(rootDir / study / f'ensemble_ranked_predictions_{dataset}set.csv')

    # order by ascending score
    rankingTblOrd = rankingTbl.sort_values('score', axis=0, ascending=True)

    # compute precision at k
    precision_at_k = {k: np.nan for k in k_arr}
    for k_i in range(len(k_arr)):
        if len(rankingTblOrd) < k_arr[k_i]:
            precision_at_k[k_arr[k_i]] = np.nan
        else:
            precision_at_k[k_arr[k_i]] = \
                np.sum(rankingTblOrd['label'][-k_arr[k_i]:]) / k_arr[k_i]

    np.save(save_dir / f'precision_at_k_{dataset}.npy', precision_at_k)

    if verbose:
        print(f'Dataset {dataset}: {precision_at_k}')

    # compute precision at k curve
    precision_at_k = {k: np.nan for k in k_curve_arr}
    for k_i in range(len(k_curve_arr)):
        if len(rankingTblOrd) < k_curve_arr[k_i]:
            precision_at_k[k_curve_arr[k_i]] = np.nan
        else:
            precision_at_k[k_curve_arr[k_i]] = \
                np.sum(rankingTblOrd['label'][-k_curve_arr[k_i]:]) / k_curve_arr[k_i]

    np.save(save_dir / f'precision_at_k_curve_{dataset}.npy', precision_at_k)

    if plot:
        # precision at k curve
        f, ax = plt.subplots()
        ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()))
        ax.set_ylabel('Precision')
        ax.set_xlabel('Top-K')
        ax.grid(True)
        ax.set_xticks(k_curve_arr_plot)
        ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
        # ax.set_ylim(top=1)
        ax.set_ylim([-0.01, 1.01])
        f.savefig(save_dir / f'precision_at_k_{dataset}.svg')
        plt.close()

        # misclassified examples at k curve
        f, ax = plt.subplots()
        kvalues = np.array(list(precision_at_k.keys()))
        precvalues = np.array(list(precision_at_k.values()))
        num_misclf_examples = kvalues - kvalues * precvalues
        ax.plot(kvalues, num_misclf_examples)
        ax.set_ylabel('Number Misclassfied TCEs')
        ax.set_xlabel('Top-K')
        ax.grid(True)
        ax.set_xticks(k_curve_arr_plot)
        ax.set_xlim([k_curve_arr[0], k_curve_arr[-1]])
        ax.set_ylim(bottom=-0.01)
        f.savefig(save_dir / f'misclassified_at_k_{dataset}.svg')
        plt.close()

#%% Compute precision at k for experiments

studies = [
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_wksnormmaxflux-wks_correctprimarygapping_ptempstat_albedostat_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_wksnormmaxflux-wks_correctprimarygapping_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_wks-selfnorm_secsymphase_correctprimarygapping_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_correctprimarygapping_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_correctprimarygapping_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_nopps',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu'
    # 'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_astronet_secsymphase_nopps_ckoiper',
    # 'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_exonet_secsymphase_nopps_ckoiper',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_tpsfeatures_tces1',
    # 'keplerdr25-tps_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper',
    'keplerdr25-tps_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_nopps_ckoiper_dvmodel'
]

datasets = ['train', 'val', 'test']

# k_arr = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
# k_arr = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # without PPs
k_arr = {'train': [100, 1000, 1268], 'val': [50, 150, 161], 'test': [50, 150, 177]}  # TCEs-1 only and without PPs

k_curve_arr = {
    # 'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
    # 'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    # 'train': np.linspace(25, 1800, 100, endpoint=True, dtype='int'),  # PPs
    # 'val': np.linspace(25, 200, 10, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
    # 'train': np.linspace(180, 1800, 10, endpoint=True, dtype='int'),  # without PPs
    # 'val': np.linspace(20, 220, 21, endpoint=True, dtype='int'),
    # 'test': np.linspace(10, 250, 25, endpoint=True, dtype='int'),
    'train': np.linspace(100, 1200, 12, endpoint=True, dtype='int'),  # without PPs and TCEs-1 only
    'val': np.linspace(20, 160, 15, endpoint=True, dtype='int'),
    'test': np.linspace(20, 170, 17, endpoint=True, dtype='int')
}
k_curve_arr_plot = {
    # 'train': np.linspace(200, 2000, 10, endpoint=True, dtype='int'),
    # 'val': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    # 'train': np.linspace(200, 1800, 8, endpoint=True, dtype='int'),  # without PPs
    # 'val': np.linspace(20, 220, 11, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 10, endpoint=True, dtype='int')
    # 'train': np.linspace(180, 1800, 10, endpoint=True, dtype='int'),
    # 'val': np.linspace(20, 220, 21, endpoint=True, dtype='int'),
    # 'test': np.linspace(10, 250, 25, endpoint=True, dtype='int'),
    'train': np.linspace(100, 1200, 12, endpoint=True, dtype='int'),  # without PPs and TCEs-1 only
    'val': np.linspace(20, 160, 15, endpoint=True, dtype='int'),
    'test': np.linspace(20, 170, 17, endpoint=True, dtype='int')
}

num_thresholds = 1000

rootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble')

for study in studies:
    print(f'Running for study {study}')
    for dataset in datasets:
        compute_precision_at_k(study, dataset, rootDir, k_arr[dataset], k_curve_arr[dataset], k_curve_arr_plot[dataset],
                               f'thr{num_thresholds}', plot=True, verbose=True)
        compute_prcurve_roc_auc(rootDir / study, dataset, num_thresholds, f'thr{num_thresholds}', plot=True)


#%% Plot ROC, PR curve and Precision at k curve for a set of studies

studies = {
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-6stellar_prelu',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar_prelu': 'ExoMiner-TPS',
    # 'keplerdr25_g2001-l201_spline_gbal_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-DV',
    ###
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar': 'Exonet',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux': 'Astronet',
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/robovetter_analysis/without_PPs/12-22-2020_2kthr': 'Robovetter',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-DV',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu': 'ExoMiner-TPS',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-PPs',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkoisnopps_starshuffle_configD_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'ExoMiner-No PPs'
    ###
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux_prelu': 'Baseline',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-loe_prelu': 'Baseline + L Odd-Even',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-lwks_prelu': 'Baseline + L Wk. Secondary',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-glcentr_std_noclip_prelu': 'Baseline + GL Centroid',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-6stellar_prelu': 'Baseline + Stellar',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-bfap_prelu': 'Baseline + Boostrap FA Prob.',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-ghost_prelu': 'Baseline + Ghost Diagnostic',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configD_glflux-rollingband_prelu': 'Baseline + Rolling Band Diagnostic',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-lwks_fluxnorm-loe-6stellar-bfap-ghost-rollingband-co_kic_oot-wksmaxmes_prelu': 'Exominer_new',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar-bfap-ghost-rollingband_prelu': 'Exominer',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_configE_glflux-glcentr_std_noclip-loe-lwks-6stellar_prelu': 'Exominer-TPS',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_astronet-300epochs-es20patience_glflux': 'Astronet',
    # 'keplerdr25_g2001-l201_spline_nongapped_norobovetterkois_starshuffle_exonet-300epochs-es20patience_glflux-glcentr_fdl-6stellar': 'Exonet'
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_wksnormmaxflux-wks_correctprimarygapping_ptempstat_albedostat_nopps': '+SecSymPhase+PrimGap+SecNormMaxFluxSec+AlbedoPlanetTempStats',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_wksnormmaxflux-wks_correctprimarygapping_nopps': '+SecSymPhase+PrimGap+SecNormalizedMaxFluxSec',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_wks-selfnorm_secsymphase_correctprimarygapping_nopps': '+SecSelfnorm+PrimGap',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_correctprimarygapping_nopps': '+SecSymPhase+PrimGap',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_correctprimarygapping_nopps': '+PrimGap',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_secsymphase_nopps': '+SecSymPhase',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu_nopps': 'Wo/ PPs',
    # 'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_prelu': 'Config K',
    'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_astronet_secsymphase_nopps_ckoiper/thr1000': 'AstroNet',
    'keplerdr25-dv_g2001-l201_9tr_spline_gapped_norobovetterkois_starshuffle_exonet_secsymphase_nopps_ckoiper/thr1000': 'ExoNet',
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per/comparison_ranking_50exoplntpaper/GPC': 'GPC',
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per/comparison_ranking_50exoplntpaper/RFC': 'RFC',
    'keplerdr25-dv_g301-l31_6tr_spline_nongapped_norobovetterkois_starshuffle_configK_secsymphase_wksnormmaxflux-wks_corrprimgap_ptempstat_albedostat_wstdepth_fwmstat_nopps_ckoiper_secparams_prad_per/thr1000': 'ExoMiner'
}
studies = {Path(key): val for key, val in studies.items()}

rootDir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble')
datasets = ['train', 'val', 'test']

# k_arr = {'train': [100, 1000, 2084], 'val': [50, 150, 257], 'test': [50, 150, 283]}
k_arr = {'train': [100, 1000, 1818], 'val': [50, 150, 222], 'test': [50, 150, 251]}  # PPs

# k_curve_arr = {
#     # 'train': np.linspace(25, 2000, 100, endpoint=True, dtype='int'),
#     # 'val': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
#     # 'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
#     'train': np.linspace(25, 1800, 100, endpoint=True, dtype='int'),  # PPs
#     'val': np.linspace(20, 220, 21, endpoint=True, dtype='int'),
#     'test': np.linspace(25, 250, 10, endpoint=True, dtype='int'),
# }
k_curve_arr_plot = {
    # 'train': np.linspace(200, 2000, 10, endpoint=True, dtype='int'),
    # 'val': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    # 'test': np.linspace(25, 250, 8, endpoint=True, dtype='int'),
    'train': np.linspace(180, 1800, 10, endpoint=True, dtype='int'),  # PPs
    'val': np.linspace(20, 220, 11, endpoint=True, dtype='int'),
    'test': np.linspace(25, 250, 10, endpoint=True, dtype='int')
}

k_arr_limits = {
    'train': [0, k_arr['train'][-1]],
    'val': [0, k_arr['val'][-1]],
    'test': [0, k_arr['val'][-1]],
}

# plot precision at k curve
for dataset in datasets:

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            precision_at_k = np.load(study /
                                     f'metrics_{dataset}set.npy',
                                     allow_pickle=True).item()['precision_at_k_curve']
        elif studyName in ['GPC', 'RFC']:
            precision_at_k = np.load(study /
                                     f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                     allow_pickle=True).item()['precision_at_k_curve']
        else:
            precision_at_k = np.load(rootDir / study / f'precision_at_k_curve_{dataset}.npy', allow_pickle=True).item()

        ax.plot(list(precision_at_k.keys()), list(precision_at_k.values()), label=studyName)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    # ax.set_xticks(np.linspace(k_arr[0], k_arr[-1], 11, endpoint=True))
    # ax.set_xticks(np.linspace(25, 250, 10, endpoint=True, dtype='int'))
    ax.set_xticks(k_curve_arr_plot[dataset])
    ax.set_yticks(np.linspace(0.9, 1, 11))
    ax.set_xlim([k_curve_arr[dataset][0], k_curve_arr[dataset][-1]])
    # ax.set_xlim(k_arr_limits[dataset])
    ax.set_ylim(top=1.001, bottom=0.9)
    ax.legend(loc=3, fontsize=7)

    f.savefig(rootDir / f'precision_at_k_{dataset}.svg')
    plt.close()

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            precision_at_k = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['precision_at_k_curve']
        elif studyName in ['GPC', 'RFC']:
            precision_at_k = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                     allow_pickle=True).item()[
                'precision_at_k_curve']
        else:
            precision_at_k = np.load(rootDir / study / f'precision_at_k_curve_{dataset}.npy', allow_pickle=True).item()

        kvalues = np.array(list(precision_at_k.keys()))
        precvalues = np.array(list(precision_at_k.values()))
        num_misclf_examples = kvalues - kvalues * precvalues
        ax.plot(kvalues, num_misclf_examples, label=studyName)

    ax.set_ylabel('Number Misclassfied TCEs')
    ax.set_xlabel('Top-K')
    ax.grid(True)
    # ax.set_xticks(np.linspace(k_arr[0], k_arr[-1], 11, endpoint=True))
    # ax.set_xticks(k_arr[dataset])
    # ax.set_yticks(np.linspace(0.9, 1, 11))
    # ax.set_xlim([k_arr[dataset][0], k_arr[dataset][-1]])
    # ax.set_ylim(top=1.001, bottom=0.9)
    ax.set_xticks(k_curve_arr_plot[dataset])
    # ax.set_xlim(k_arr_limits[dataset])
    ax.set_xlim([k_curve_arr_plot[dataset][0], k_curve_arr_plot[dataset][-1]])
    ax.legend(loc=2, fontsize=7)

    f.savefig(rootDir / f'misclassified_at_k_{dataset}.svg')
    plt.close()


# plot PR curve
for dataset in datasets:

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['recal_thr']
            precisionThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['precision_thr']
        elif studyName in ['GPC', 'RFC']:
            recallThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                allow_pickle=True).item()['recall_thr']
            precisionThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                   allow_pickle=True).item()['precision_thr']
        else:
            # metricsStudy = np.load(rootDir / study / 'results_ensemble.npy', allow_pickle=True).item()
            # recallThr = metricsStudy[f'{dataset}_rec_thr']
            # precisionThr = metricsStudy[f'{dataset}_prec_thr']
            metricsStudy = np.load(rootDir / study / f'metrics_{dataset}.npy', allow_pickle=True).item()
            recallThr = metricsStudy['rec_thr']
            precisionThr = metricsStudy['prec_thr']

        ax.plot(recallThr, precisionThr, label=studyName)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.grid(True)
    ax.set_xticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    ax.legend(loc=3, fontsize=7)

    f.savefig(rootDir / f'precision_recall_curve_{dataset}.svg')
    plt.close()

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['recal_thr']
            precisionThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['precision_thr']
        elif studyName in ['GPC', 'RFC']:
            recallThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                allow_pickle=True).item()['recall_thr']
            precisionThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                   allow_pickle=True).item()['precision_thr']
        else:
            # metricsStudy = np.load(rootDir / study / 'results_ensemble.npy', allow_pickle=True).item()
            # recallThr = metricsStudy[f'{dataset}_rec_thr']
            # precisionThr = metricsStudy[f'{dataset}_prec_thr']
            metricsStudy = np.load(rootDir / study / f'metrics_{dataset}.npy', allow_pickle=True).item()
            recallThr = metricsStudy['rec_thr']
            precisionThr = metricsStudy['prec_thr']

        ax.plot(recallThr, precisionThr, label=studyName)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.grid(True)
    ax.set_xticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.5, 1, 11, endpoint=True))
    ax.set_xlim([0.85, 1])
    ax.set_ylim([0.7, 1])
    ax.legend(loc=3, fontsize=7)

    f.savefig(rootDir / f'precision_recall_curve_{dataset}_zoomin.svg')
    plt.close()

# plot ROC
for dataset in datasets:

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['recal_thr']
            fprThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['fpr_thr']
        elif studyName in ['GPC', 'RFC']:
            recallThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy', allow_pickle=True).item()['recall_thr']
            fprThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy', allow_pickle=True).item()['fpr_thr']
        else:
            # metricsStudy = np.load(rootDir / study / 'results_ensemble.npy', allow_pickle=True).item()
            # recallThr = metricsStudy[f'{dataset}_rec_thr']
            # fprThr = metricsStudy[f'{dataset}_fp'] / (metricsStudy[f'{dataset}_tn'] + metricsStudy[f'{dataset}_fp'])
            metricsStudy = np.load(rootDir / study / f'metrics_{dataset}.npy', allow_pickle=True).item()
            recallThr = metricsStudy['rec_thr']
            fprThr = metricsStudy['fpr_thr']

        ax.plot(fprThr, recallThr, label=studyName)

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.grid(True)
    ax.set_xticks(np.linspace(0, 0.5, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.75, 1, 11, endpoint=True))
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.75, 1])
    ax.legend(loc=4, fontsize=7)

    f.savefig(rootDir / f'roc_{dataset}.svg')
    plt.close()

    f, ax = plt.subplots()

    for study, studyName in studies.items():

        if studyName == 'Robovetter':
            recallThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['recal_thr']
            fprThr = np.load(study / f'metrics_{dataset}set.npy', allow_pickle=True).item()['fpr_thr']
        elif studyName in ['GPC', 'RFC']:
            recallThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                                allow_pickle=True).item()['recall_thr']
            fprThr = np.load(study / f'metrics_PP_{studyName}_final_{dataset}set.npy',
                             allow_pickle=True).item()['fpr_thr']
        else:
            # metricsStudy = np.load(rootDir / study / 'results_ensemble.npy', allow_pickle=True).item()
            # recallThr = metricsStudy[f'{dataset}_rec_thr']
            # fprThr = metricsStudy[f'{dataset}_fp'] / (metricsStudy[f'{dataset}_tn'] + metricsStudy[f'{dataset}_fp'])
            metricsStudy = np.load(rootDir / study / f'metrics_{dataset}.npy', allow_pickle=True).item()
            recallThr = metricsStudy['rec_thr']
            fprThr = metricsStudy['fpr_thr']

        ax.plot(fprThr, recallThr, label=studyName)

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.grid(True)
    ax.set_xticks(np.linspace(0, 0.5, 11, endpoint=True))
    ax.set_yticks(np.linspace(0.75, 1, 11, endpoint=True))
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.95, 1])
    ax.legend(loc=4, fontsize=7)

    f.savefig(rootDir / f'roc_{dataset}_zoomin.svg')
    plt.close()
