

#%% Add KOI fields to ranking

rankingTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                            'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/ranked_predictions_testset',
                            header=0)

keplerTceTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/koi_ephemeris_matching/'
                           'q1_q17_dr25_tce_2019.03.12_updt_tcert_extendedtceparams_updt_normstellarparamswitherrors_koi_processed.csv',
                           header=0)

# fields to be added to the ranking table
addedFields = ['kepoi_name', 'kepler_name', 'koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
               'koi_fpflag_ec', 'koi_comment', 'koi_datalink_dvr', 'koi_datalink_dvs']

# instantiate these fields as NaN for all TCEs
rankingTceTbl = pd.concat([rankingTceTbl, pd.DataFrame(columns=addedFields)])

for tce_i, tce in rankingTceTbl.iterrows():
    rankingTceTbl.iloc[tce_i, addedFields] = keplerTceTbl.loc[(keplerTceTbl['target_id'] == tce.kepid) &
                                                              (keplerTceTbl['tce_plnt_num'] == tce.tce_n)][addedFields]

rankingTceTbl.to_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/'
                     'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/ranked_predictions_testset_koi',
                     index=False)