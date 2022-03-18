""" Get RUWE values for KICs from Gaia data releases. """

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

#%%  Get ruwe from Gaia DR2-KIC table in "Revised Radii of Kepler Stars and Planets Using Gaia Data Release 2"

root_dir = Path(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/kepler_gaia_ruwe/ruwe-gaiadr2_1-4-2022/')
root_dir.mkdir(exist_ok=True)

res_dir = root_dir / f'gaiadr2_2020_q1q17dr25_kics_{datetime.now().strftime("%m-%d-%Y_%H%M")}'
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='ruwe')
logger_handler = logging.FileHandler(filename=res_dir / f'ruwe_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

# tce_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
#                   'ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
# tce_tbl = pd.read_csv(tce_tbl_fp)
kic_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv')
kic_tbl = pd.read_csv(kic_tbl_fp, usecols=['kepid']).rename(columns={'kepid': 'target_id'})
logger.info(f'Using KIC table: {str(kic_tbl_fp)}')

gaia_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_dr_tbls/gaia_dr2_2020/GKSPCPapTable1_Final.txt')
gaia_tbl = pd.read_csv(gaia_tbl_fp,
                       sep='&',
                       lineterminator='\\')
logger.info(f'Using GAIA DR2 catalog: {str(gaia_tbl_fp)}')
gaia_tbl.rename(columns={'KIC': 'target_id'}, inplace=True)
logger.info(f'Number of KICs in the KIC-Gaia DR2 catalog: {len(gaia_tbl)}')

kic_ruwe_tbl = kic_tbl.merge(gaia_tbl[['target_id', 'RUWE']], on=['target_id'], how='left',
                             validate='many_to_one').drop_duplicates(subset='target_id')
kic_ruwe_tbl = kic_ruwe_tbl.rename(columns={'RUWE': 'ruwe'})

# filtering for planets only validated by us                             ]
# kic_ruwe_tbl_valplnt = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'].drop_duplicates(subset='target_id')
# kic_ruwe_tbl_valplnt[['target_id', 'ruwe', 'validated_by']].to_csv(res_dir / 'kics_validated_plnts.csv', index=False)

# kic_ruwe_tbl[['target_id', 'ruwe', 'validated_by']].to_csv(res_dir / 'kics.csv', index=False)
kic_ruwe_tbl.to_csv(res_dir / 'kics.csv', index=False)
logger.info('Finished extracting RUWE values from the KIC-Gaia DR2 catalog.')
logger.info(f'Number of KICs with missing RUWE: {kic_ruwe_tbl["ruwe"].isna().sum()} out of {len(kic_ruwe_tbl)}  KICs.')

#%% Get Gaia source ids for KICs from different tables and use them to get RUWE values from Gaia data releases

query_gaia_dr = 'gaiadr2'  # 'gaia_dr2' or 'gaia_edr3'
src_tbl = 'gaiadr2_2018'  # 'simbad',  'gaiadr2_2018', 'crossmatch_gaiadr2'

res_dir = Path(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/'
               f'kepler_gaia_ruwe/src-{src_tbl}_ruwe-{query_gaia_dr}_keplerq1q17dr25_kics_{datetime.now().strftime("%m-%d-%Y_%H%M")}')
res_dir.mkdir(exist_ok=True)

# set up logger
logger = logging.getLogger(name='ruwe')
logger_handler = logging.FileHandler(filename=res_dir / f'ruwe_run.log', mode='w')
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info(f'Starting run...')

# tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
# kic_tbl = tce_tbl.drop_duplicates(subset='target_id')[['target_id']]
# val_kics = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'][['target_id']].drop_duplicates(subset='target_id')

logger.info(f'Using as source table of source ids for KICs: {src_tbl}')

# get Gaia source ids from paper "Revised Radii of Kepler Stars and Planets Using Gaia Data Release 2" for KICs
if src_tbl == 'gaiadr2_2018':
    gaia_dr2_2018_tbl_fp = Path(
        '/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/gaia_dr_tbls/gaia_dr2_2018/DR2PapTable1.txt')
    logger.info(f'Loading KIC-Gaia source id table from: {str(gaia_dr2_2018_tbl_fp)}')
    gaia_dr2_2018_tbl = pd.read_csv(gaia_dr2_2018_tbl_fp,
                                    sep='&',
                                    lineterminator='\n',
                                    dtype={'source_id': str,
                                           'KIC': str}
                                    )
    # gaia_dr2_2020_tbl = pd.read_csv(gaia_dr2_2020_tbl_fp, dtype={'source_id': np.int64})
    gaia_dr2_2018_tbl = gaia_dr2_2018_tbl.loc[~gaia_dr2_2018_tbl['source_id'].isna()]
    gaia_dr2_2018_tbl = gaia_dr2_2018_tbl.astype(dtype={'source_id': np.int64})
    gaia_dr2_2018_tbl.rename(columns={'KIC': 'target_id'}, inplace=True)
    # gaia_dr2_2018_tbl_val = gaia_dr2_2018_tbll.loc[gaia_dr2_2018_tbl['target_id'].isin(val_kics['target_id'])]
    source_ids_kic_tbl = gaia_dr2_2018_tbl

# get Gaia source ids from SIMBAD for validated KICs
elif src_tbl == 'simbad':
    simbad_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/'
                         'data/ephemeris_tables/kepler/kic_catalogs/gaia/simbad_allkics.csv')
    simbad_tbl = pd.read_csv(simbad_tbl_fp,
                             dtype={'source_id': str})
    logger.info(f'Loading KIC-Gaia source id table from: {str(simbad_tbl_fp)}')
    simbad_tbl = simbad_tbl.loc[~simbad_tbl['source_id'].isna()]
    simbad_tbl = simbad_tbl.astype({'source_id': np.int64})
    # simbad_tbl.to_csv(res_dir / 'simbad_kics.csv', index=False)
    simbad_tbl = simbad_tbl.rename(columns={'kic': 'target_id'})
    source_ids_kic_tbl = simbad_tbl

# get Gaia DR2 source ids from cross-match for validated KICs
if src_tbl == 'crossmatch_gaiadr2':
    cross_match_gaiadr2_kepler_fp = Path('/Users/msaragoc/'
                                         'OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
                                         'ephemeris_tables/kepler/kic_catalogs/gaia/cross_match_gaiadr2_kepler.csv')
    cross_match_gaiadr2_kepler = pd.read_csv(cross_match_gaiadr2_kepler_fp)
    logger.info(f'Loading KIC-Gaia source id table from: {str(cross_match_gaiadr2_kepler_fp)}')
    cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.merge(
        cross_match_gaiadr2_kepler[['kepid']].value_counts().to_frame('n_occur').reset_index().rename(
            columns={'index': 'kepid'})[['kepid', 'n_occur']],
        on=['kepid'], how='left', validate='many_to_one')
    # select matches for which the angular distance between KIC and Gaia are smaller
    cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.sort_values('kepler_gaia_ang_dist').drop_duplicates('kepid', keep='first')
    # cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.groupby('kepid')['kepler_gaia_ang_dist'].min().reset_index()
    # cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.loc[(cross_match_gaiadr2_kepler['kepid'].isin(val_kics['target_id']))]
    # cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.loc[(cross_match_gaiadr2_kepler['kepid'].isin(validated_targets['target_id'])) & (cross_match_gaiadr2_kepler['n_occur'] == 1)]
    cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.rename(columns={'kepid': 'target_id'})
    # cross_match_gaiadr2_kepler[['source_id', 'target_id', 'n_occur', 'kepler_gaia_ang_dist']].to_csv(res_dir / 'cross_match_gaiadr2_kics.csv', index=False)
    source_ids_kic_tbl = cross_match_gaiadr2_kepler

assert len(source_ids_kic_tbl) == len(source_ids_kic_tbl['target_id'].unique())

logger.info(f'Number of KICs: {len(source_ids_kic_tbl)}.')
logger.info(f'Number of repeated source IDs: {(source_ids_kic_tbl["source_id"].value_counts() > 1).sum()}')
logger.info(f'Number of unique source IDs: {len(source_ids_kic_tbl["source_id"].unique())}.')

# count number of occurrences for each source id
source_ids_cnts = \
    source_ids_kic_tbl['source_id'].value_counts().to_frame('n_occur_source_ids').reset_index().rename(
        columns={'index': 'source_id'})
source_ids_kic_tbl = source_ids_kic_tbl.merge(source_ids_cnts,
                                              on=['source_id'],
                                              how='left',
                                              validate='many_to_one')

# select the objects to query using their source id
source_ids_tbl = Table.from_pandas(source_ids_kic_tbl[['source_id']])
source_ids_tbl.to_pandas().to_csv(res_dir / 'kics_sourceids_fromtbl.csv', index=False)
source_ids_votbl = from_table(source_ids_tbl)
source_id_tbl_fp = res_dir / 'kics_sourceids.xml'
source_ids_votbl.to_xml(str(source_id_tbl_fp))

upload_resource = source_id_tbl_fp
upload_tbl_name = 'kics_sourceids'
output_fp = res_dir / f'{query_gaia_dr}_kics.csv'

logger.info(f'Querying {query_gaia_dr} source ids for their RUWE values...')
# query
# query = f"select g.source_id, g.ruwe from gaiaedr3.gaia_source as g, tap_upload.{upload_tbl_name} as f where g.source_id = f.source_id"
# query = f"select g.source_id from gaiadr2.gaia_source as g, tap_upload.{upload_tbl_name} as f where g.source_id = f.source_id"
if query_gaia_dr == 'gaiaedr3':
    query = f"SELECT g.source_id, g.ruwe FROM gaiaedr3.gaia_source as g JOIN tap_upload.{upload_tbl_name} as f ON g.source_id = f.source_id"
elif query_gaia_dr == 'gaiadr2':
    query = f"SELECT g.source_id, g.ruwe FROM gaiadr2.ruwe as g JOIN tap_upload.{upload_tbl_name} " \
            f"as f ON g.source_id = f.source_id"
logger.info(f'Query performed: {query}')
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

logger.info(f'Query finished and results saved to {str(output_fp)}.')

ruwe_tbl = pd.read_csv(output_fp)
# ruwe_tbl = ruwe_tbl.merge(source_ids_kic_tbl[['target_id', 'source_id']], on=['source_id'], how='left',
#                           validate='one_to_one')
# ruwe_tbl.to_csv(res_dir / f'{output_fp.stem}_with_kicid.csv', index=False)

# in case there are duplicate source ids in the KIC-source id table, many_to_one
if (source_ids_kic_tbl["source_id"].value_counts() > 1).sum() > 0:
    kic_ruwe_tbl = source_ids_kic_tbl[['target_id', 'source_id', 'n_occur_source_ids']].merge(
        ruwe_tbl.drop_duplicates('source_id'),
        on=['source_id'],
        how='left',
        validate='many_to_one')
else:
    kic_ruwe_tbl = source_ids_kic_tbl[['target_id', 'source_id', 'n_occur_source_ids']].merge(
        ruwe_tbl.drop_duplicates('source_id'),
        on=['source_id'],
        how='left',
        validate='one_to_one')

kic_ruwe_tbl.to_csv(res_dir / f'{output_fp.stem}_with_kicid.csv', index=False)
logger.info(f'Number of KICs without RUWE value: {kic_ruwe_tbl["ruwe"].isna().sum()} ouf of {len(kic_ruwe_tbl)}')

#%% plotting RUWE

res_dir = Path(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_gaia_ruwe/ruwe-gaiaedr3_12-30-2021/gaiadr2_2020_q1q17dr25_kics_01-27-2022_1158')
# logger.info('Loading table with KICs to be queried in Gaia...')
kic_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv',
                      usecols=['kepid']).rename(columns={'kepid': 'target_id'})
assert len(kic_tbl) == len(kic_tbl['target_id'].unique())
# logger.info(f'{len(kic_tbl)} KICs in KIC table.')

tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
val_kics = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'][['target_id']].drop_duplicates(subset='target_id')

ruwe_tbl = pd.read_csv(res_dir / 'kics.csv')

ruwe_source = 'Gaia DR2-KIC 2020'  # 'Gaia DR2-KIC 2018', 'Gaia DR2 KIC cross-match Megan Bedell', 'SIMBAD'

# source_tbl_title = {
#     '': 'Gaia DR2'
# }

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

# %% Combine RUWE values from different sources of matches between KICs and Gaia source ids

tce_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')

# kic_tbl = tce_tbl.drop_duplicates(subset='target_id')[['target_id']]
kic_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv')
kic_tbl = pd.read_csv(kic_tbl_fp, usecols=['kepid']).rename(columns={'kepid': 'target_id'})

val_kics = tce_tbl.loc[tce_tbl['validated_by'] == 'us_exominer_2021'][['target_id']].drop_duplicates(subset='target_id')

kic_tbl['validated_by'] = np.nan
kic_tbl.loc[kic_tbl['target_id'].isin(val_kics['target_id']), 'validated_by'] = 'us_exominer_2021'

gaia_dr2_2018_ruwe_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/kepler_gaia_ruwe/ruwe-gaiadr2_1-4-2022/src-gaiadr2_2018_ruwe-gaiadr2_keplerq1q17dr25_kics_01-04-2022_1354/gaiadr2_kics_with_kicid.csv')
gaia_dr2_2018_ruwe_tbl = gaia_dr2_2018_ruwe_tbl[['target_id', 'ruwe']].rename(
    columns={'source_id': 'source_id_gaiadr2_2018', 'ruwe': 'ruwe_gaiadr2_2018'})

gaia_dr2_2020_ruwe_tbl = pd.read_csv(
    '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/kepler_gaia_ruwe/ruwe-gaiaedr3_12-30-2021/gaiadr2_paper_all_kic_12-30-2021_0730/gaia_edr3_kics_with_kicid.csv')
gaia_dr2_2020_ruwe_tbl = gaia_dr2_2020_ruwe_tbl.rename(
    columns={'source_id': 'source_id_gaiadr2_2020', 'ruwe': 'ruwe_gaiadr2_2020'})

# simbad_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/simbad_all_kic_12-30-2021_0709/gaia_edr3_kics_with_kicid.csv')
# simbad_tbl = simbad_tbl.rename(columns={'source_id':  'source_id_simbad', 'ruwe': 'ruwe_simbad'})

crossmatch_ruwe_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/kepler_gaia_ruwe/ruwe-gaiadr2_1-4-2022/src-crossmatch_gaiadr2_ruwe-gaiadr2_keplerq1q17dr25_kics_01-04-2022_1405/gaiadr2_kics_with_kicid.csv')
crossmatch_ruwe_tbl = crossmatch_ruwe_tbl.rename(
    columns={'source_id': 'source_id_crossmatch', 'ruwe': 'ruwe_crossmatch'})

match_tbl = kic_tbl

# add RUWE values from RUWE source tables
ruwe_src_tbls = [
    gaia_dr2_2018_ruwe_tbl,
    gaia_dr2_2020_ruwe_tbl,
    # simbad_tbl,
    crossmatch_ruwe_tbl
]
for tbl in ruwe_src_tbls:
    match_tbl = match_tbl.merge(tbl, on=['target_id'], how='left', validate='one_to_one')

# choose final ruwe value using preference rules for the sources
match_tbl['ruwe_final'] = np.nan
match_tbl['ruwe_final_source'] = np.nan
for kic_i, kic in match_tbl.iterrows():

    if not np.isnan(kic['ruwe_gaiadr2_2018']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_gaiadr2_2018'], 'ruwe_gaiadr2_2018']
    elif not np.isnan(kic['ruwe_gaiadr2_2020']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_gaiadr2_2020'], 'ruwe_gaiadr2_2020']
    # elif not np.isnan(kic['ruwe_simbad']):
    #     match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_simbad'], 'simbad']
    elif not np.isnan(kic['ruwe_crossmatch']):
        match_tbl.loc[kic_i, ['ruwe_final', 'ruwe_final_source']] = [kic['ruwe_crossmatch'], 'crossmatch']

res_dir = Path('/data5/tess_project/Data/Ephemeris_tables/Kepler/kepler_gaia_ruwe/ruwe-gaiadr2_1-4-2022')

match_tbl_val = match_tbl.loc[match_tbl['validated_by'] == 'us_exominer_2021']
bins = np.linspace(0, 2, 100)  # np.logspace(0, 2, 100)

ruwe_thr = 1.4

#  plot histogram of RUWE for population of KICs with validated planets by us and the whole Kepler dataset
f, ax = plt.subplots()
ax.hist(match_tbl_val['ruwe_final'], bins=bins, edgecolor='k', label='KICs with validated planets', zorder=2)
ax.hist(match_tbl['ruwe_final'], bins=bins, edgecolor='k', label='All KICs in dataset', zorder=1)
# ax[1].hist(match_tbl['ruwe_final'], bins=bins, edgecolor='k', cumulative=True, density=True)
ax.set_xlabel('RUWE')
ax.set_ylabel('Target counts')
# ax[1].set_ylabel('Normalized cumulative \ncounts')
ax.set_xlim([0, 2])
# ax[0].set_xlim([0, 2])
ax.set_title(f'KICs with RUWE > {ruwe_thr} = {(match_tbl_val["ruwe_final"] > ruwe_thr).sum()}\nRUWE source: multiple')
ax.set_yscale('log')
# ax.set_xscale('log')
# plt.show(
f.savefig(res_dir / 'hist_ruwe_all_kics.png')

# add number of validated planets by us per KIC
count_val = tce_tbl[['target_id', 'tce_plnt_num', 'validated_by']].copy()
count_val['cnt_validated_by'] = 0
count_val.loc[count_val['validated_by'] == 'us_exominer_2021', 'cnt_validated_by'] = 1
num_val_plnt_per_kic = count_val.groupby('target_id').sum()[['cnt_validated_by']].reset_index()
match_tbl = match_tbl.merge(num_val_plnt_per_kic, on=['target_id'], how='left', validate='one_to_one')
match_tbl.to_csv(res_dir / 'all_kics_ruwe_multsources.csv', index=False)

# %% Add RUWE values to the TCE table

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17_DR25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc_modelchisqr.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

ruwe_tbl = pd.read_csv(
    '/data5/tess_project/Data/Ephemeris_tables/Kepler/kepler_gaia_ruwe/ruwe-gaiadr2_1-4-2022/all_kics_ruwe_multsources.csv')
ruwe_tbl.rename(columns={'ruwe_final': 'ruwe', 'ruwe_final_source': 'ruwe_source'}, inplace=True)

tce_tbl = tce_tbl.merge(ruwe_tbl[['target_id', 'ruwe', 'ruwe_source']], on=['target_id'], validate='many_to_one')

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ruwe.csv', index=False)
