""" Getting RUWE values for KICs associated with validated planets. """

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from pathlib import Path
from datetime import datetime
import logging
from astropy.table import Table
from astropy.io.votable import from_table
from astroquery.simbad import Simbad

#%%

root_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe')
root_dir.mkdir(exist_ok=True)

res_dir = root_dir / f'gaiadr2_all_kic_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='ruwe')
logger_handler = logging.FileHandler(filename=res_dir / f'ruwe_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

#%% Get ruwe from Gaia DR2-KIC table in "Revised Radii of Kepler Stars and Planets Using Gaia Data Release 2"

tce_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)
logger.info(f'Using ranking: {str(tce_tbl_fp)}')

gaia_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
                   'ephemeris_tables/kepler/kic_catalogs/gaia/gaia_dr2_2020/GKSPCPapTable1_Final.txt')
gaia_tbl = pd.read_csv(gaia_tbl_fp,
                       sep='&',
                       lineterminator='\\')
logger.info(f'Using GAIA DR2 catalog: {str(gaia_tbl_fp)}')
gaia_tbl.rename(columns={'KIC': 'target_id'}, inplace=True)

kic_ruwe_tbl = tce_tbl.merge(gaia_tbl[['target_id', 'RUWE']], on=['target_id'], how='left', validate='many_to_one').drop_duplicates(subset='target_id')
kic_ruwe_tbl = kic_ruwe_tbl.rename(columns={'RUWE': 'ruwe'})

# filtering for planets only validated by us                             ]
# kic_ruwe_tbl_valplnt = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'].drop_duplicates(subset='target_id')
# kic_ruwe_tbl_valplnt[['target_id', 'ruwe', 'validated_by']].to_csv(res_dir / 'kics_validated_plnts.csv', index=False)

kic_ruwe_tbl[['target_id', 'ruwe', 'validated_by']].to_csv(res_dir / 'kics.csv', index=False)

#%% Get ruwe values from different sources for  matching KICs to source ids in  Gaia DR2 and then using them  in Gaia EDR3

res_dir = Path(f'/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/crossmatch_gaiadr2_all_kic_{datetime.now().strftime("%m-%d-%Y_%H%M")}')
res_dir.mkdir(exist_ok=True)

tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
kic_tbl = tce_tbl.drop_duplicates(subset='target_id')[['target_id']]
val_kics = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'][['target_id']].drop_duplicates(subset='target_id')

# get Gaia source ids from paper "Revised Radii of Kepler Stars and Planets Using Gaia Data Release 2" for KICs
# gaia_dr2_2020_tbl = pd.read_csv(
#     '/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/kic_catalogs/gaia/gaia_dr2_2018/DR2PapTable1.txt',
#     sep='&',
#     lineterminator='\\',
#     dtype={'source_id': str}
#     )
# gaia_dr2_2020_tbl = gaia_dr2_2020_tbl.loc[~gaia_dr2_2020_tbl['source_id'].isna()]
# gaia_dr2_2020_tbl = gaia_dr2_2020_tbl.astype(dtype={'source_id': np.int64})
# gaia_dr2_2020_tbl.rename(columns={'KIC': 'target_id'}, inplace=True)
# # gaia_dr2_2020_tbl_val = gaia_dr2_2020_tbl.loc[gaia_dr2_2020_tbl['target_id'].isin(val_kics['target_id'])]

# # get Gaia source ids from SIMBAD for validated KICs
# simbad_tbl = pd.read_csv('/Users/msaragoc/Downloads/simbad_allkics.csv', dtype={'gaia dr2': str})
# simbad_tbl = simbad_tbl.loc[~simbad_tbl['gaia dr2'].isna()]
# simbad_tbl = simbad_tbl.astype({'gaia dr2': np.int64})
# simbad_tbl.to_csv(res_dir / 'simbad_val_kics.csv', index=False)
# simbad_tbl = simbad_tbl.rename(columns={'gaia dr2': 'source_id'})

# get Gaia DR2 source ids from cross-match for validated KICs
cross_match_gaiadr2_kepler = pd.read_csv('/Users/msaragoc/Downloads/cross_match_gaiadr2_kepler.csv')
cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.merge(
    cross_match_gaiadr2_kepler[['kepid']].value_counts().to_frame('n_occur').reset_index().rename(columns={'index': 'kepid'})[['kepid', 'n_occur']],
    on=['kepid'], how='left', validate='many_to_one')
# select matches for which the angular distance between KIC and  Gaia are smaller
cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.sort_values('kepler_gaia_ang_dist').drop_duplicates('kepid', keep='first')
# cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.groupby('kepid')['kepler_gaia_ang_dist'].min().reset_index()
# cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.loc[(cross_match_gaiadr2_kepler['kepid'].isin(val_kics['target_id']))]
# cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.loc[(cross_match_gaiadr2_kepler['kepid'].isin(validated_targets['target_id'])) & (cross_match_gaiadr2_kepler['n_occur'] == 1)]
cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.rename(columns={'kepid': 'target_id'})
cross_match_gaiadr2_kepler[['source_id', 'target_id', 'n_occur', 'kepler_gaia_ang_dist']].to_csv(res_dir / 'cross_match_gaiadr2_kics.csv', index=False)

# select the objects to query
source_ids_kic_tbl = cross_match_gaiadr2_kepler
source_ids_selected = source_ids_kic_tbl[['source_id']]

source_ids_tbl = Table.from_pandas(source_ids_selected)
source_ids_tbl.to_pandas().to_csv(res_dir / 'kics_sourceids_fromtbl.csv', index=False)
source_ids_votbl = from_table(source_ids_tbl)
source_id_tbl_fp = res_dir / 'kics_sourceids.xml'
source_ids_votbl.to_xml(str(source_id_tbl_fp))

upload_resource = source_id_tbl_fp
upload_tbl_name = 'kics_sourceids'
output_fp = res_dir / 'gaia_edr3_kics.csv'

# query
# query = f"select g.source_id, g.ruwe from gaiaedr3.gaia_source as g, tap_upload.{upload_tbl_name} as f where g.source_id = f.source_id"
# query = f"select g.source_id from gaiadr2.gaia_source as g, tap_upload.{upload_tbl_name} as f where g.source_id = f.source_id"
query = f"SELECT g.source_id, g.ruwe FROM gaiaedr3.gaia_source as g JOIN tap_upload.{upload_tbl_name} as f ON g.source_id = f.source_id"

# j = Gaia.launch_job(query=query,
#                     upload_resource=str(upload_resource),
#                     upload_table_name=upload_tbl_name,
#                     verbose=True,
#                     output_file=str(output_fp),
#                     dump_to_file=True,
#                     output_format='csv',
#                     )
j = Gaia.launch_job_async(query=query,
                          upload_resource=str(upload_resource),
                          upload_table_name=upload_tbl_name,
                          verbose=True,
                          output_file=str(output_fp),
                          dump_to_file=True,
                          output_format='csv',
                          )

r = j.get_results()
r.pprint()

ruwe_tbl = pd.read_csv(output_fp)
# ruwe_tbl = ruwe_tbl.merge(source_ids_kic_tbl[['target_id', 'source_id']], on=['source_id'], how='left',
#                           validate='one_to_one')
# in case there are duplicate source ids in the KIC-source id table
kic_ruwe_tbl = source_ids_kic_tbl[['target_id', 'source_id']].merge(ruwe_tbl.drop_duplicates('source_id'),
                                                         on=['source_id'], how='left', validate='many_to_one')
kic_ruwe_tbl.to_csv(res_dir / f'{output_fp.stem}_with_kicid.csv', index=False)

#%% plotting RUWE

tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
kic_tbl = tce_tbl.drop_duplicates(subset='target_id')
val_kics = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'][['target_id']].drop_duplicates(subset='target_id')

ruwe_tbl, ruwe_source = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/crossmatch_gaiadr2_all_kic_12-30-2021_0521/gaia_edr3_kics_with_kicid.csv'), \
                        'Crossmatch Gaia DR2 KIC'
ruwe_thr = 1.4
ruwe_tbl_valkics = ruwe_tbl.loc[ruwe_tbl['target_id'].isin(val_kics['target_id'])]

bins = np.linspace(0, 2, 100)  # np.logspace(0, 2, 100)

f, ax = plt.subplots(2, 1)
ax[0].hist(ruwe_tbl_valkics['ruwe'], bins=bins, edgecolor='k')
ax[1].hist(ruwe_tbl_valkics['ruwe'], bins=bins, edgecolor='k', cumulative=True, density=True)
ax[1].set_xlabel('RUWE')
ax[0].set_ylabel('Target counts')
ax[1].set_ylabel('Normalized cumulative \ncounts')
ax[1].set_xlim([0, 2])
ax[0].set_xlim([0, 2])
ax[0].set_title(f'{len(ruwe_tbl_valkics)}/{len(val_kics)} KICs for validated planets (targets with missing RUWE: '
                f'{ruwe_tbl_valkics["ruwe"].isna().sum()})\n RUWE source: {ruwe_source}\n Targets with RUWE > {ruwe_thr} = {(ruwe_tbl_valkics["ruwe"] > ruwe_thr).sum()}')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.show(
f.tight_layout()
f.savefig(res_dir / 'hist_ruwe_validated_plnts_kics.png')

f, ax = plt.subplots()
ax.hist(ruwe_tbl_valkics['ruwe'], bins=bins, edgecolor='k', label='KICs with validated planets', zorder=2)
ax.hist(ruwe_tbl['ruwe'], bins=bins, edgecolor='k',
        label='All KICs in dataset', zorder=1)
ax.set_xlabel('RUWE')
ax.set_ylabel('Target counts')
ax.legend()
ax.set_title(f'Validated targets vs all targets RUWE \n RUWE source: {ruwe_source}')
ax.set_yscale('log')
ax.set_xlim([0, 2])
# ax.set_xscale('log')
# plt.show()
f.savefig(res_dir / 'hist_ruwe_validated_plnts_kics_vs_allkics.png')

#%% Combine RUWE values from different sources of matches between KICs and Gaia source ids

ranking_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)
ranking_tbl_valplnt = ranking_tbl.loc[ranking_tbl['validated_by'] == 'us_exominer_2021']
validated_targets = ranking_tbl_valplnt.drop_duplicates(subset='target_id')[['target_id']]

kic_ruwe_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/12-20-2021_1018/kics_validated_plnts.csv')
kic_ruwe_tbl = kic_ruwe_tbl[['target_id', 'RUWE']].rename(columns={'RUWE':  'ruwe_kictbl'})

gaia_dr2_paper_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/gaiadr2_paper_12-29-2021_0951/gaia_edr3_val_kics_with_kicid.csv')
gaia_dr2_paper_tbl = gaia_dr2_paper_tbl.rename(columns={'source_id':  'source_id_gaiadr2paper', 'ruwe': 'ruwe_gaiadr2paper'})

simbad_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/simbad_12-29-2021_1006/gaia_edr3_val_kics.csv')
simbad_tbl = simbad_tbl.rename(columns={'source_id':  'source_id_simbad', 'ruwe': 'ruwe_simbad'})

crossmatch_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/crossmatch_gaiadr2_kepler_12-29-2021_1339/gaia_edr3_val_kics.csv')
crossmatch_tbl = crossmatch_tbl.rename(columns={'source_id':  'source_id_crossmatch', 'ruwe': 'ruwe_crossmatch'})

match_tbl = validated_targets
for tbl in [kic_ruwe_tbl, gaia_dr2_paper_tbl, simbad_tbl, crossmatch_tbl]:
    match_tbl = match_tbl.merge(tbl, on=['target_id'], how='left', validate='one_to_one')

# choose final ruwe value using prefence rules for the sources
match_tbl['ruwe_final'] = np.nan
match_tbl['ruwe_final_source'] = np.nan
for kic_i, kic in match_tbl.iterrows():

    if not np.isnan(kic['ruwe_kictbl']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_kictbl'], 'kictbl']
    elif not np.isnan(kic['ruwe_gaiadr2paper']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_gaiadr2paper'], 'gaiadr2paper']
    elif not np.isnan(kic['ruwe_simbad']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_simbad'], 'simbad']
    elif not np.isnan(kic['ruwe_crossmatch']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_crossmatch'], 'crossmatch']

res_dir = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/validated_kics_combinedsources_12-30-2021/')
match_tbl.to_csv(res_dir / 'val_kics_ruwe_multsources.csv', index=False)

bins = np.linspace(0, 2, 100)  # np.logspace(0, 2, 100)

f, ax = plt.subplots(2, 1)
ax[0].hist(match_tbl['ruwe_final'], bins=bins, edgecolor='k')
ax[1].hist(match_tbl['ruwe_final'], bins=bins, edgecolor='k', cumulative=True, density=True)
ax[1].set_xlabel('RUWE')
ax[0].set_ylabel('Target counts')
ax[1].set_ylabel('Normalized cumulative \ncounts')
ax[1].set_xlim([0, 2])
ax[0].set_xlim([0, 2])
ax[0].set_title(f'{len(validated_targets)} KICs for validated planets (targets with missing RUWE: '
                f'{match_tbl["ruwe_final"].isna().sum()})\n RUWE source: multiple')
ax[0].set_yscale('log')
# ax.set_xscale('log')
# plt.show(
f.savefig(res_dir / 'hist_ruwe_validated_plnts_kics.png')
