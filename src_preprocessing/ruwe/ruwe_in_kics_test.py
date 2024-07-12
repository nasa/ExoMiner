""" Utility and test code for main script. """

# 3rd party
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from pathlib import Path
from datetime import datetime
from astropy.table import Table
from astroquery.simbad import Simbad
# from astroquery.utils.tap.core import TapPlus
from astropy.io.votable import from_table

#%% get all KICs in Q1-Q17 DR25

kics = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/kic_catalogs/q1_q17_dr25_stellar_plus_supp.csv', usecols=['kepid'])
assert len(kics['kepid'].unique()) == len(kics)
kics.to_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/all_kics_q1q17dr25_1-4-2022.csv', index=False)

#%% getting KICs for validated planets by us

# gaia = TapPlus(url="http://gea.esac.esa.int/tap-server/tap")

ranking_tbl_fp = Path('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)
ranking_tbl_valplnt = ranking_tbl.loc[ranking_tbl['validated_by'] == 'us_exominer_2021']
validated_targets = ranking_tbl_valplnt.drop_duplicates(subset='target_id')

#%% querying SIMBAD for object IDs (KIC, Gaia DR2 and EDR3 source id)

# tce_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/q1-q17_dr25/11-17-2021_1243/q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_cpkoiperiod_rba_cnt0n_valpc.csv')
# kic_tbl = tce_tbl.drop_duplicates(subset='target_id')[['target_id']]
kic_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv',
                      usecols=['kepid'])

data_to_tbl = {'kic': [], 'gaia dr2': [], 'gaia edr3': []}

# result_table = Simbad.query_object("KIC 1026957")

# for kic in validated_targets['target_id']:
kic = 11413698
for kic_i, kic in enumerate(kic_tbl.to_numpy().flatten()):

    if (kic_i + 1) % 1000 == 0:
        print(f'Querying SIMBAD for KIC {kic} ({kic_i + 1}/{len(kic_tbl)})')

    aaaa
    result_table = Simbad.query_objectids(f"KIC {kic}")

    data_to_tbl['kic'].append(kic)
    data_to_tbl['gaia dr2'].append(np.nan)
    data_to_tbl['gaia edr3'].append(np.nan)

    if result_table is None:
        continue

    result_table = result_table.to_pandas()
    result_table = result_table['ID'].str.decode('utf-8')
    result_table = result_table.loc[result_table.str.contains('Gaia')]

    for id_obj in result_table:
        if 'Gaia EDR3' in id_obj:
            data_to_tbl['gaia edr3'][-1] = id_obj.split(' ')[-1]
        elif 'Gaia DR2' in id_obj:
            data_to_tbl['gaia dr2'][-1] = id_obj.split(' ')[-1]

data_df = pd.DataFrame(data_to_tbl)

data_df['source_id'] = np.nan
data_df['source_id_release'] = np.nan
for kic_i, kic in data_df.iterrows():
    if isinstance(kic['gaia edr3'], str):  # not np.isnan(kic['gaia edr3']):
        data_df.loc[kic_i, ['source_id', 'source_id_release']] = [kic['gaia edr3'], 'gaia edr3']

    if isinstance(kic['gaia dr2'], str):  # not np.isnan(kic['gaia dr2']):
        data_df.loc[kic_i, ['source_id', 'source_id_release']] = [kic['gaia dr2'], 'gaia dr2']

data_df.to_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/'
               'kepler/kic_catalogs/gaia/simbad_kics_q1q17dr25.csv', index=False)

#%% get 1'' arcsec search radius for crossmatch between KICs and Gaia DR2 source ids from Megan Bedell

data = Table.read('/Users/msaragoc/Downloads/kepler_dr2_1arcsec.fits', format='fits')
cross_match_gaiadr2_kepler = data.to_pandas()
cross_match_gaiadr2_kepler.to_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/ephemeris_tables/kepler/kic_catalogs/gaia/cross_match_gaiadr2_kepler.csv', index=False)

#%% Testing queries to Gaia

# Gaia.query_object()
query = "select top 100 * "\
        "from gaiaedr3.gaia_source order by source_id"
query = "select top 100 source_id, ruwe from gaiaedr3.gaia_source where source_id > 4282339439322680000" #  where exists {select 1 from gaiaedr3.gaia_source where source_id=source_id}"
# query = "select top 100 source_id, ruwe from gaiaedr3.gaia_source where exists (select 1 from gaiaedr3.gaia_source where source_id >= source_id)"
job = Gaia.launch_job_async(query,
                            output_file='/Users/msaragoc/Downloads/gaia_edr3_query-tbl.csv',
                            dump_to_file=True,
                            output_format='csv')

#%% Query one object at a time from Gaia

gaia_dr2_2020_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/'
                                'ephemeris_tables/kepler/kic_catalogs/gaia/gaia_dr2_2018/DR2PapTable1.csv')
res_dir = Path(f'/Users/msaragoc/Downloads/gaia_edr3_queries/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
# if res_dir.exists():
#     res_dir = Path(str(res_dir) + )
res_dir.mkdir(exist_ok=True)
source_ids = gaia_dr2_2020_tbl['source_id'][:3]
for object_i, source_id in enumerate(source_ids):
    # source_id  = 4120088258008531456
    source_id_str = f'{source_id:.0f}'
    # aa
    # source_id_str = '4120088258008531456'
    print(f'Querying object source id {source_id_str} ({object_i + 1} out of {len(gaia_dr2_2020_tbl["source_id"])})')
    query = f"select source_id, ruwe from gaiaedr3.gaia_source where source_id = {source_id_str}"
    job = Gaia.launch_job_async(query,
                                output_file=res_dir / f'gaia_edr3_{source_id_str}.csv',
                                dump_to_file=True,
                                output_format='csv',
                                verbose=True)

#%% Testing querying Gaia DR2 RUWE table

res_dir = Path('/Users/msaragoc/Downloads/')

# get Gaia DR2 source ids from cross-match for validated KICs
cross_match_gaiadr2_kepler = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/'
                                         'exoplanet_transit_classification/data/ephemeris_tables/kepler/kic_catalogs/'
                                         'gaia/cross_match_gaiadr2_kepler.csv')
cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.merge(
    cross_match_gaiadr2_kepler[['kepid']].value_counts().to_frame('n_occur').reset_index().rename(
        columns={'index': 'kepid'})[['kepid', 'n_occur']],
    on=['kepid'], how='left', validate='many_to_one')
# select matches for which the angular distance between KIC and  Gaia are smaller
cross_match_gaiadr2_kepler = \
    cross_match_gaiadr2_kepler.sort_values('kepler_gaia_ang_dist').drop_duplicates('kepid', keep='first')
cross_match_gaiadr2_kepler = cross_match_gaiadr2_kepler.rename(columns={'kepid': 'target_id'})

# select the objects to query using their source id
source_ids_kic_tbl = cross_match_gaiadr2_kepler
source_ids_selected = source_ids_kic_tbl[['source_id']][:3]

source_ids_tbl = Table.from_pandas(source_ids_selected)
source_ids_tbl.to_pandas().to_csv(res_dir / 'kics_sourceids_fromtbl.csv', index=False)
source_ids_votbl = from_table(source_ids_tbl)
source_id_tbl_fp = res_dir / 'kics_sourceids.xml'
source_ids_votbl.to_xml(str(source_id_tbl_fp))

upload_resource = source_id_tbl_fp
upload_tbl_name = 'kics_sourceids'
output_fp = res_dir / 'gaia_edr3_kics.csv'

# query
# query = f"SELECT g.source_id, g.ruwe FROM gaiaedr3.gaia_source as g JOIN tap_upload.{upload_tbl_name} " \
#         f"as f ON g.source_id = f.source_id"
query = f"SELECT g.source_id, g.ruwe FROM gaiadr2.ruwe as g JOIN tap_upload.{upload_tbl_name} " \
        f"as f ON g.source_id = f.source_id"

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

#%%  Count number of occurrences of source ids and add them to the source id table


def _count_occurr(row, tbl):
    # print('row', row)
    # print((tbl['source_id'] == 2076065630440124928).sum())
    # aaa
    return (tbl['source_id'] == 2076065630440124928).sum()

source_ids_cnts = \
    source_ids_kic_tbl['source_id'].value_counts().to_frame('n_occur_source_ids').reset_index().rename(
        columns={'index': 'source_id'})
source_ids_kic_tbl = source_ids_kic_tbl.merge(source_ids_cnts,
                                            on=['source_id'],
                                            how='left',
                                            validate='many_to_one')
# source_ids_kic_tbl['n_occur_source_ids'] = source_ids_kic_tbl.apply(_count_occurr, args=(b,), axis=1)

#%% Add RUWE values to ranking table

ranking_tbl_fp = Path('/Users/msaragoc/Downloads/ranking_comparison_with_paper_12-18-2020_merged_ra_dec_prad_CV_v15_targetcnts.csv')
ranking_tbl = pd.read_csv(ranking_tbl_fp)
ruwe_tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/Analysis/kepler_gaia_ruwe/all_kics_ruwe_multsources_12-30-2021.csv')
ranking_tbl = ranking_tbl.merge(ruwe_tbl[['target_id', 'ruwe_final']],
                                            on=['target_id'],
                                            how='left',
                                            validate='many_to_one')
ranking_tbl.to_csv(ranking_tbl_fp.parent / f'{ranking_tbl_fp.stem}_ruwe_1-4-2022.csv', index=False)
