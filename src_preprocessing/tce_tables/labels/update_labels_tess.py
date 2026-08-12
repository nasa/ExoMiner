"""
Update labels for TESS SPOC TCEs using ephemeris matching and other results against a set of catalogs of objects with
dispositions.

Priority rules are followed to come up with the most comprehensive and clean set of labels possible.
"""

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np


def load_tce_tbl(tce_tbl_fp):
    """Load TCE table.

    :param Path tce_tbl_fp: filepath to TCE table
    :return pd DataFrame: TCE table
    """
    
    # load TCE table
    tce_tbl = pd.read_csv(tce_tbl_fp)

    # drop columns in TCE table that were there from previous labeling process
    drop_columns = [
        'in_tso_spoc_ebs',
        'matched_toiexofop',
        'match_corr_coef_toiexofop',
        'TFOPWG Disposition',
        'TESS Disposition',
        'Period (days)',
        'Duration (hours)',
        'Depth (ppm)',
        'Epoch (BJD)',
        'Planet Radius (R_Earth)',
        'Planet SNR',
        'Spectroscopy Observations',
        'Imaging Observations',
        'Time Series Observations',
        'Comments',
        'matched_astronet-qlp_tce',
        'match_corr_coef_astronet-qlp_tce',
        'label_astronet-qlp',
        'matched_villanova_ebs',
        'match_corr_coef_villanova_ebs',
        'tec_fluxtriage_pass',
        'tec_fluxtriage_comment',
        'matched_object',
        'sg1_master_disp',
        'exofop_toi',
        'exofop_toi_match_score',
        'Master Disposition',
        'Phot Disposition',
        'Spec Disposition',
        'sg_gaia_ruwe',
        'sg_comments',
    ]
    
    tce_tbl = tce_tbl.drop(columns=drop_columns, errors='ignore')
    
    return tce_tbl
    

def add_exofop_toi_matches(match_toi_tbl_fp, toi_tbl_fp, tce_tbl, update=False):
    """Add ExoFOP TOI matches to TCE table.

    :param str match_toi_tbl_fp: filepath to match table between TCEs and ExoFOP TOIs
    :param str toi_tbl_fp: filepath to ExoFOP TOI table
    :param pd DataFrame tce_tbl: TCE table
    :param bool updated: if True, it will update TCE table by overwriting existing ExoFOP TOI match columns; if False, it will merge results as if it had no existing ExoFOP TOI matches
    :return pd DataFrame: TCE table with ExoFOP TOI matches
    """
    
    # define columns that want to be added from the TOI catalog
    toi_cols = [
        'uid',
        'TFOPWG Disposition',
        'TESS Disposition',
        'period',  # 'Period (days)',
        'duration',  # 'Duration (hours)',
        'Depth (ppm)',
        'epoch',  # 'Epoch (BJD)',
        'Planet Radius (R_Earth)',
        'Planet SNR',
        'Spectroscopy Observations',
        'Imaging Observations',
        'Time Series Observations',
        'Comments',
    ]

    # load TCE-ExoFOP TOI matching table
    match_tbl = pd.read_csv(match_toi_tbl_fp)
    # rename columns in match table
    match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_toiexofop',
                                        'match_corr_coef': 'match_corr_coef_toiexofop'})
    
    # load ExoFOP TOI table used in matching TCEs with TOIs
    toi_tbl = pd.read_csv(toi_tbl_fp, header=0, usecols=toi_cols)
    # rename columns in TOI table
    toi_tbl = toi_tbl.rename(columns={'uid': 'matched_toiexofop', 'epoch': 'epoch_exofop',
                                    'duration': 'duration_exofop', 'period': 'period_exofop'})
    
    match_tbl_wtois = match_tbl.merge(toi_tbl, on='matched_toiexofop', how='left', validate='many_to_one')
    
    if update:
        tce_tbl.set_index('uid', inplace=True)
        match_tbl_wtois.set_index('uid', inplace=True)
        tce_tbl.update(match_tbl_wtois)
        tce_tbl.reset_index(inplace=True)
    else:
        drop_cols = [col for col in toi_cols if col not in ['uid', 'epoch_exofop', 'duration_exofop', 'period_exofop']] + ['matched_toiexofop', 'match_corr_coef_toiexofop']
        tce_tbl = tce_tbl.drop(columns=drop_cols, errors='ignore')
        # merge matching results to TCE table
        tce_tbl = tce_tbl.merge(match_tbl_wtois, on='uid', how='left', validate='one_to_one')

    print(f'TCE TFOP WG disposition counts after ExoFOP TOI matching:\n{tce_tbl["TFOPWG Disposition"].value_counts()}')
    print(f'TCE TESS disposition counts after ExoFOP TOI matching:\n{tce_tbl["TESS Disposition"].value_counts()}')
    
    return tce_tbl


def add_prsa_ebs_matches(match_tbl_fp, prsa_ebs_tbl_fp, tce_tbl):
    """Add Prsa's EB matches to TCE table.

    :param str match_tbl_fp: filepath to match table between TCEs and Prsa's EBs
    :param str prsa_ebs_tbl_fp: filepath to Prsa's EB catalog
    :param pd DataFrame tce_tbl: TCE table
    :return pd DataFrame: TCE table with Prsa's EB matches
    """
     
     # load TCE-Prsa's EBs matching table
    match_tbl = pd.read_csv(match_tbl_fp)
    match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_villanova_ebs',
                                        'match_corr_coef': 'match_corr_coef_villanova_ebs'})
    # merge matching results to TCE table
    tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

    # define columns that want to be added to the TCE table
    prsa_eb_tbl_cols = [
        'uid',
    ]
    # load Villanova's EBs table used in matching with TCEs
    prsa_ebs_tbl = pd.read_csv(prsa_ebs_tbl_fp, usecols=prsa_eb_tbl_cols)
    prsa_ebs_tbl = prsa_ebs_tbl.rename(columns={'uid': 'matched_villanova_ebs'})
    # merge tce table with toi table based on matching
    tce_tbl = tce_tbl.merge(prsa_ebs_tbl, on='matched_villanova_ebs', how='left', validate='many_to_one')

    print(f'TCEs matched to Prsa\'s EBs:\n{(~tce_tbl["matched_villanova_ebs"].isna()).sum()}')

    return tce_tbl


def add_kostov_ebs_matches(match_tbl_fp, kostov_ebs_tbl_fp, tce_tbl):
    """Add Kostov's EB matches to TCE table.

    :param str match_tbl_fp: filepath to match table between TCEs and Prsa's EBs
    :param str kostov_ebs_tbl_fp: filepath to Kostov's EB catalog
    :param pd DataFrame tce_tbl: TCE table
    :return pd DataFrame: TCE table with Kostov's EB matches
    """
    
    # load TCE-Kostovs's EBs matching table
    match_tbl = pd.read_csv(match_tbl_fp)
    match_tbl = match_tbl.rename(columns={'signal_a': 'uid', 'signal_b': 'matched_kostov_ebs',
                                        'match_corr_coef': 'match_corr_coef_kostov_ebs'})
    # merge matching results to TCE table
    tce_tbl = tce_tbl.merge(match_tbl, on='uid', how='left', validate='one_to_one')

    # define columns that want to be added to the TCE table
    kostov_ebs_tbl_cols = [
        'uid',
    ]
    # load Kostov's EBs table used in matching with TCEs
    kostov_ebs_tbl = pd.read_csv(kostov_ebs_tbl_fp, usecols=kostov_ebs_tbl_cols)
    kostov_ebs_tbl = kostov_ebs_tbl.rename(columns={'uid': 'matched_kostov_ebs'})
    # merge tce table with toi table based on matching
    tce_tbl = tce_tbl.merge(kostov_ebs_tbl, on='matched_kostov_ebs', how='left', validate='many_to_one')

    print(f'TCEs matched to Kostov\'s EBs:\n{(~tce_tbl["matched_kostov_ebs"].isna()).sum()}')

    return tce_tbl


def add_tec_ntp_matches(tec_tbl_fp, tce_tbl, update=False):
    """Add TEC's flux triage-based NTP matches to TCE table.

    :param str tec_tbl_fp: filepath to TEC flux triage's catalog
    :param pd DataFrame tce_tbl: TCE table
    :param bool update: if True, will update TCE table with the TEC flux triage results based on uid index; if False, 
        will merge the TEC flux triage results to the TCE table based on uid and add new columns to the TCE table for the TEC flux triage results
    :return pd DataFrame: TCE table with TEC flux triage NTP matches
    """
    
    tec_cols = [
        'uid',
        'tec_fluxtriage_pass',
        'tec_fluxtriage_comment',
    ]
    tec_tbl = pd.read_csv(tec_tbl_fp, usecols=tec_cols)

    if update:
        tce_tbl.set_index('uid', inplace=True)
        tec_tbl.set_index('uid', inplace=True)
        tce_tbl.update(tec_tbl)
        tce_tbl.reset_index(inplace=True)
    else:
        tce_tbl = tce_tbl.drop(columns=['tec_fluxtriage_pass', 'tec_fluxtriage_comment'], errors='ignore')
        # merge matching results to TCE table
        tce_tbl = tce_tbl.merge(tec_tbl[tec_cols], on='uid', how='left', validate='one_to_one')

    n_tces_found_in_tec_flux_triage = tce_tbl["tec_fluxtriage_pass"].notna().sum()
    print(f'Number of TCEs found in TEC flux triage: {n_tces_found_in_tec_flux_triage}')
    
    return tce_tbl


def add_sg1_data(sg1_toi_tbl_fp, tce_tbl):
    """Add SG1 TOI catalog data to the TCE table based on ExoFOP TOI matches.

    :param str sg1_toi_tbl_fp: filepath to SG1 catalog
    :param pd DataFrame tce_tbl: TCE table
    :return pd DataFrame: TCE table with SG1 TOI data
    """
    
    sg1_toi_cols = [
    'TOI',
    'Master Disposition',
    'Phot Disposition',
    'Spec Disposition',
    'Comments',
    'SG2 Notes',
    ]
    sg1_toi_tbl = pd.read_csv(sg1_toi_tbl_fp, usecols=sg1_toi_cols)
    sg1_toi_tbl.rename(columns={'Master Disposition': 'sg1_master_disp', 'TOI': 'matched_object',
                                'Phot Disposition': 'sg1_phot_disp', 'Spec Disposition': 'sg1_spec_disp',
                                'Comments': 'sg1_comments', 'SG2 Notes': 'sg2_notes'}, inplace=True,
                    errors='raise')
    sg1_toi_tbl['matched_object'] = sg1_toi_tbl['matched_object'].astype('str')
    sg1_toi_tbl.drop_duplicates('matched_object', inplace=True)

    tce_tbl['matched_object'] = tce_tbl['matched_object'].astype('str')

    tce_tbl = tce_tbl.merge(sg1_toi_tbl[['matched_object', 'sg1_master_disp']], how='left', on='matched_object',
                            validate='many_to_one')

    # # update labels to NEB, NPC and BD
    # idxs_tces = tce_tbl['sg1_master_disp'].isin(['BD', 'NEB', 'NPC'])
    # tce_tbl.loc[idxs_tces, 'label'] = tce_tbl.loc[idxs_tces, 'sg1_master_disp']
    # tce_tbl.loc[idxs_tces, 'label_source'] = 'SG1'
    
    return tce_tbl
    
    
def assign_labels_using_matching_information(tce_tbl, aux_tbls_names_lst):
    """Assign labels to TCEs based on matches to different disposition source catalogs. Labels are set to column `label`. 
    Column `label_source` shows which catalog was used to source the label; `matched_object` shows which object the TCE matched to from 
    the corresponding source catalog.

    :param pd DataFrame tce_tbl: TCE table
    :param list aux_tbls_names_lst: list of tables used for sourcing labels
    :return pd DataFrame: TCE table with assigned labels for TCEs
    """
    
    # initialize labels as UNK and no label source
    tce_tbl['label'] = 'UNK'
    tce_tbl['label_source'] = 'None'
    tce_tbl['matched_object'] = 'None'
    
    exofop_toi_col = 'TFOPWG Disposition'
    valid_exofop_toi_disps = ['KP', 'CP', 'FP']
    idxs_matched_exofop =  ~pd.Series(data=np.ones(len(tce_tbl), dtype='bool'))
    # 1) TFOPWG dispositions from ExoFOP TOI catalog
    if 'tois_matches' in aux_tbls_names_lst:
        if {'matched_toiexofop', 'TFOPWG Disposition'}.issubset(tce_tbl.columns):
            idxs_matched_exofop = ~tce_tbl['matched_toiexofop'].isna()
            idxs_matched_exofop_valid = ((idxs_matched_exofop) & 
                                        (tce_tbl['label'] == 'UNK') &
                                        tce_tbl[exofop_toi_col].isin(valid_exofop_toi_disps))
            tce_tbl.loc[idxs_matched_exofop, 'label_source'] = exofop_toi_col
            # only assign label to the TCEs with valid label dispositions
            tce_tbl.loc[idxs_matched_exofop_valid, 'label'] = tce_tbl.loc[idxs_matched_exofop_valid, exofop_toi_col]
            tce_tbl.loc[idxs_matched_exofop, 'matched_object'] = tce_tbl.loc[idxs_matched_exofop, 'matched_toiexofop']
        else:
            idxs_matched_exofop =  ~pd.Series(data=np.ones(len(tce_tbl), dtype='bool'))
        

    if 'kostov_ebs_matches' in aux_tbls_names_lst:
        idxs_matched_kostovebs = ((~tce_tbl['matched_kostov_ebs'].isna()) &
                                    (tce_tbl['label'] == 'UNK') &
                                    ~idxs_matched_exofop
                                    )
        tce_tbl.loc[idxs_matched_kostovebs, ['label_source']] = 'Kostov'
        tce_tbl.loc[idxs_matched_kostovebs, 'label'] = 'EB'
        tce_tbl.loc[idxs_matched_kostovebs, 'matched_object'] = (tce_tbl.loc)[idxs_matched_kostovebs, 'matched_kostov_ebs']
        
    # 2) Prsa's EBs
    if 'prsa_ebs_matches' in aux_tbls_names_lst:
        idxs_matched_villanovaebs = ((~tce_tbl['matched_villanova_ebs'].isna()) &
                                    (tce_tbl['label'] == 'UNK') &
                                    ~idxs_matched_exofop
                                    )
        tce_tbl.loc[idxs_matched_villanovaebs, ['label_source']] = 'Villanova'
        tce_tbl.loc[idxs_matched_villanovaebs, 'label'] = 'EB'
        tce_tbl.loc[idxs_matched_villanovaebs, 'matched_object'] = (tce_tbl.loc)[idxs_matched_villanovaebs, 'matched_villanova_ebs']

    # 3) create NTPs based on TEC flux triage; don't include TCEs detected as secondaries of other TCEs
    if 'tec_ntps_fluxtriage' in aux_tbls_names_lst:
        # for TESS SPOC 2-min TCEs
        idxs_matched_tec_ntps = ((tce_tbl['tec_fluxtriage_pass'] == 0) &
                                (~tce_tbl['tec_fluxtriage_comment'].str.contains('SecondaryOfPN', na=False)) &
                                ~idxs_matched_exofop &
                                (tce_tbl['label'] == 'UNK')
                                )
        tce_tbl.loc[idxs_matched_tec_ntps, ['label', 'label_source']] = 'NTP', 'TEC flux triage'
        # set to UNK those TCEs that did not pass the TEC flux triage because they failed AltDet and their period is less or
        # equal to 0.3 days
        tce_tbl.loc[(idxs_matched_exofop) &
                    (tce_tbl['label'] == 'NTP') &
                    (tce_tbl['tec_fluxtriage_comment'] == 'AltDetFail') &
                    (tce_tbl['tce_period'] <= 0.3),
                    ['label', 'label_source']] = 'UNK', 'None'

        # # for TESS SPOC FFI TCEs, match to 2-min NTP TCEs
        # idxs_matched_tec_ntps = ((~tce_tbl['matched_tecntps'].isna()) &
        #                          (~tce_tbl['matched_toiexofop'].isna()))
        # tce_tbl.loc[idxs_matched_tec_ntps, ['label', 'label_source']] = 'NTP', 'TEC flux triage'
    
    return tce_tbl

def set_secondaries_to_unlabeled_tces(tce_tbl, label_cols_to_update=['label', 'label_source']):
    """Set TCEs identified as secondaries based on shorter/longer period tests to unlabeled. TCE table must contain boolean column
        "is_secondary". TCEs with this flag True will have their "label" set to 'UNK' and "label_source" set to 'None'.

    :param pd.DataFrame tce_tbl: TCE table
    :param list label_cols_to_update: list of label columns to update
    :return pd.DataFrame: updated TCE table with secondaries set to UNK
    """
    
    tce_tbl.loc[tce_tbl['is_secondary'] == True, label_cols_to_update] = 'UNK', 'None'
    
    return tce_tbl


def main(tce_tbl_fp, new_tbl_fp, aux_tbls):
    """Main function used to load TCE table, add matches to TCEs from the source catalogs (using the matching tables), and then assign 
    labels to TCEs based on those matches. Labels are set to column `label`. Column `label_source` shows which catalog was used to 
    source the label; `matched_object` shows which object the TCE matched to from the corresponding source catalog.
    
    `aux_tbls` must contain, for each catalog used for labels, 1) the catalog itself, and 2) a corresponding match table that resulted from
    conducting ephemeris matching (or some sort of matching) between the TCEs and the objects in the source catalog.

    :param pd DataFrame tce_tbl: TCE table
    :param Path new_tbl_fp: filepath to save new TCE table with labels
    :param dict aux_tbls: tables used for sourcing labels for the TCEs in the TCE table
    """
    
    #% Load TCE table
    print(f'Loading TCE table from {tce_tbl_fp}...')    
    tce_tbl = load_tce_tbl(tce_tbl_fp)

    #% Add dispositions from ExoFOP TOI catalog based on ephemeris matching
    if 'tois' and 'tois_matches' in aux_tbls:
        print('Adding matching information for ExoFOP TOIs')
        toi_tbl_fp = aux_tbls['tois']
        match_toi_tbl_fp = aux_tbls['tois_matches']

        tce_tbl = add_exofop_toi_matches(match_toi_tbl_fp, toi_tbl_fp, tce_tbl)

    #% Add dispositions from Kostov's EB based on ephemeris matching
    if 'kostov_ebs' and 'kostov_ebs_matches' in aux_tbls:
        print('Adding matching information for Kostov EBs')
        kostov_ebs_tbl_fp = aux_tbls['kostov_ebs']
        match_tbl_fp = aux_tbls['kostov_ebs_matches']

        tce_tbl = add_kostov_ebs_matches(match_tbl_fp, kostov_ebs_tbl_fp, tce_tbl)

    #% Add dispositions from Prsa's EB based on ephemeris matching
    if 'prsa_ebs' and 'prsa_ebs_matches' in aux_tbls:
        print('Adding matching information for Prsa EBs')
        prsa_ebs_tbl_fp = aux_tbls['prsa_ebs']
        match_tbl_fp = aux_tbls['prsa_ebs_matches']

        tce_tbl = add_prsa_ebs_matches(match_tbl_fp, prsa_ebs_tbl_fp, tce_tbl)

    #% Add dispositions from TEC flux triage
    if 'tec_ntps_fluxtriage' in aux_tbls:
        print('Adding matching information for TEC flux triage results')
        tec_tbl_fp =aux_tbls['tec_ntps_fluxtriage']

        tce_tbl = add_tec_ntp_matches(tec_tbl_fp, tce_tbl)

    #% Set labels of TCEs according to priority rules
    print('Assigning labels to TCEs based on matching results')
    tce_tbl = assign_labels_using_matching_information(tce_tbl, list(aux_tbls.keys()))
    
    if 'is_secondary' in tce_tbl:
        tce_tbl = set_secondaries_to_unlabeled_tces(tce_tbl)

    #% add SG1 dispositions
    if 'sg1' in aux_tbls:
        print('Adding information from SG1 catalog based on TOI assignments')
        sg1_toi_tbl_fp = aux_tbls['sg1']

        tce_tbl = add_sg1_data(sg1_toi_tbl_fp, tce_tbl)

    print(f'TCE disposition counts after update:\n {tce_tbl["label"].value_counts()}')
    print(f'TCE label source counts after update:\n {tce_tbl["label_source"].value_counts()}')

    print(f'Saving new TCE table to {new_tbl_fp}...')
    tce_tbl.to_csv(new_tbl_fp, index=False)
    
    print('Done.')


if __name__ == '__main__':
    
    tce_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_ffi/tess-spoc-ffi-tces-dv_s73-s81_3-20-2026_0951/tess-spoc-ffi-tces-dv_s73-s81_3-20-2026_0951_stellartic8_ruwegaiadr2_preproc.csv')
    new_tbl_fp = tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_label.csv'
    
    aux_tbls = {
        'tois': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/exofop_tois/tois_3-2-2026_processed_ephem_matching.csv',
        'tois_matches': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/ephemeris-matching_tces-spoc_ffi-tces-dv_s73-s81_exofop-tois_3-30-2026_1501/matched_signals_thr0.75.csv',
        'kostov_ebs': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/ebs_kostov_7.9k_catalog.csv',
        'kostov_ebs_matches': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/ephemeris-matching_tces-spoc_ffi-tces-dv_s73-s81_kostov-ebs_3-30-2026_1547/matched_signals_thr0.75.csv',
        'prsa_ebs': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat_processed_with-duration.csv',
        'prsa_ebs_matches': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/ephemeris-matching_tces-spoc_ffi-tces-dv_s73-s81_prsa-ebs_3-30-2026_1144/matched_signals_thr0.75.csv',
        # 'tec_ntps_fluxtriage': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tec/tec_spoc_2min_fluxtriage_s1-s41_10-4-2023.csv',
        'sg1': '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/sg1/sg1_tois_4-21-2025_1118.csv'
    }
    
    main(tce_tbl_fp, new_tbl_fp, aux_tbls)
    