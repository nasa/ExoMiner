""" Plot figure of views and scalar features for examples in a given TFRecord dataset. """

# 3rd party
import tensorflow as tf
from pathlib import Path
import numpy as np

# local
from src_preprocessing.lc_preprocessing.utils_manipulate_tfrecords import plot_features_example

# list of examples to plot figures of preprocessed data (views + scalar parameters)
# examples_ids = ['6500206-2']  # Kepler (target_id, tce_plnt_num)
examples_ids = ['390651552-1-S23']  # TESS (target_id, tce_plnt_num, Ssector_run)
mission = 'tess'  # `kepler` or `tess`

scheme = (3, 5)  # plot placement in a nxm matrix in the figure; must be consistent with number of views
basename = 'all_views'  # basename for figures

num_scalar_features_per_line = 6  # number of scalar features per line

# TFRecord directory
# tfrecDir = Path('/data5/tess_project/Data/tfrecords/Kepler/Q1-Q17_DR25/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_data/tfrecordskeplerdr25-dv_g301-l31_spline_nongapped_newvalpcs_tessfeaturesadjs_12-1-2021_experiment-normalized')
tfrecDir = Path(
    '/data5/tess_project/Data/tfrecords/TESS/tfrecordstessS1S40-tces_dv_g301-l31_5tr_spline_nongapped_allts-allscalars_1-10-2022_0704_data/tfrecordstessS1S40-tces_dv_g301-l31_5tr_spline_nongapped_allts-allscalars_1-10-2022_0704')

# plot directory
plotDir = Path('/home/msaragoc/Downloads')
plotDir.mkdir(exist_ok=True)

# get filepaths to TFRecord files
tfrecFiles = [file for file in tfrecDir.iterdir() if 'shard' in file.stem and not file.suffix == '.csv']

# set views to be plotted
views = [
    'global_flux_view',
    'local_flux_view',
    'global_flux_view_fluxnorm',
    'local_flux_view_fluxnorm',
    # 'global_flux_odd_view',
    'local_flux_odd_view',
    'local_flux_odd_view_fluxnorm',
    # 'global_flux_even_view',
    'local_flux_even_view',
    'local_flux_even_view_fluxnorm',
    # 'local_flux_oddeven_view_diff',
    # 'local_flux_oddeven_view_diff_dir',
    # 'global_weak_secondary_view',
    'local_weak_secondary_view',
    # 'local_weak_secondary_view_selfnorm',
    # 'local_weak_secondary_view_fluxnorm',
    'local_weak_secondary_view_max_flux-wks_norm',
    # centroid
    'global_centr_view',
    'local_centr_view',
    # 'global_centr_view_std_clip',
    # 'local_centr_view_std_clip',
    'global_centr_view_std_noclip',
    'local_centr_view_std_noclip',
    # 'global_centr_view_medind_std',
    # 'local_centr_view_medind_std',
    # 'global_centr_view_medcmaxn',
    # 'local_centr_view_medcmaxn',
    # 'global_centr_view_medcmaxn_dir',
    # 'local_centr_view_medcmaxn_dir',
    # 'global_centr_view_medn',
    # 'local_centr_view_medn',
    # 'global_centr_fdl_view',
    # 'local_centr_fdl_view',
    # 'global_centr_fdl_view_norm',
    # 'local_centr_fdl_view_norm',
]

# set scalar parameter values to be extracted; norm: get normalized value as well; type: data type; disp: how to display value
scalarParams = {
    # stellar parameters
    'tce_steff': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_slogg': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_smet': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_sradius': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_smass': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_sdens': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'mag': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'mag_cat': {'norm': False, 'type': 'int', 'disp': 'reg'},
    # secondary
    'wst_depth': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_maxmes': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_albedo_stat': {'norm': False, 'type': 'float', 'disp': 'reg'},
    # 'tce_albedo': {'norm': False, 'type': 'int', 'disp': 'reg'},
    'tce_ptemp_stat': {'norm': False, 'type': 'float', 'disp': 'reg'},
    # 'tce_ptemp': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # 'wst_robstat': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # odd-even
    # 'tce_bin_oedp_stat': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # other parameters
    'boot_fap': {'norm': False, 'type': 'float', 'disp': 'exp'},
    'tce_cap_stat': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_hap_stat': {'norm': False, 'type': 'float', 'disp': 'reg'},
    # 'tce_cap_hap_stat_diff': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # 'tce_rb_tcount0': {'norm': False, 'type': 'int', 'disp': 'reg'},
    # 'tce_rb_tcount0n': {'norm': False, 'type': 'float', 'disp': 'reg'},
    # centroid
    # 'tce_fwm_stat': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_dikco_msky': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_dikco_msky_err': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_dicco_msky': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_dicco_msky_err': {'norm': False, 'type': 'float', 'disp': 'reg'},
    # flux
    # 'tce_max_mult_ev': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # 'tce_depth_err': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # 'tce_duration_err': {'norm': True, 'type': 'float', 'disp': 'reg'},
    # 'tce_period_err': {'norm': True, 'type': 'float', 'disp': 'reg'},
    'transit_depth': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_prad': {'norm': False, 'type': 'float', 'disp': 'reg'},
    'tce_period': {'norm': False, 'type': 'float', 'disp': 'reg'},
}

# iterate through TFRecord dataset to get data for the examples
for tfrecFile in tfrecFiles:
    tfrecord_dataset = tf.data.TFRecordDataset(str(tfrecFile))

    for string_record in tfrecord_dataset.as_numpy_iterator():

        example = tf.train.Example()
        example.ParseFromString(string_record)

        if mission == 'tess':
            target_id_example = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num_example = example.features.feature['tce_plnt_num'].int64_list.value[0]
            sector_run_example = example.features.feature['sector_run'].bytes_list.value[0].decode("utf-8")
            example_id_str = f'{target_id_example}-{tce_plnt_num_example}-S{sector_run_example}'
        elif mission == 'kepler':
            target_id_example = example.features.feature['target_id'].int64_list.value[0]
            tce_plnt_num_example = example.features.feature['tce_plnt_num'].int64_list.value[0]
            example_id_str = f'{target_id_example}-{tce_plnt_num_example}'
        else:
            raise ValueError('Mission not recognized.')

        if example_id_str not in examples_ids:
            continue

        # get label
        example_label = example.features.feature['label'].bytes_list.value[0].decode("utf-8")

        # get scalar parameters
        scalarParamsStr = ''
        for scalarParam_i, (scalarParam_name, scalarParam_opt) in enumerate(scalarParams.items()):

            if scalarParam_opt['norm']:  # get normalized feature
                scalar_param_val_norm = example.features.feature[f'{scalarParam_name}_norm'].float_list.value[0]

            if scalarParam_opt['type'] == 'int':  # integer features
                scalar_param_val = example.features.feature[scalarParam_name].int64_list.value[0]
            else:  # float features
                scalar_param_val = example.features.feature[scalarParam_name].float_list.value[0]

            if scalarParam_i % num_scalar_features_per_line == 0 and scalarParam_i != 0:
                scalarParamsStr += '\n'

            if scalarParam_opt['norm']:
                if scalarParam_opt['disp'] == 'exp':  # exp value
                    scalarParamsStr += f'{scalarParam_name}=' \
                                       f'{scalar_param_val_norm:.4f} ({scalar_param_val:.4E})|'
                elif scalarParam_opt['disp'] == 'reg':  # integer value
                    scalarParamsStr += f'{scalarParam_name}=' \
                                       f'{scalar_param_val_norm:.4f} ({scalar_param_val})|'
                else:  # float value
                    scalarParamsStr += f'{scalarParam_name}=' \
                                       f'{scalar_param_val_norm:.4f} ({scalar_param_val:.4f})|'
            else:
                if scalarParam_opt['disp'] == 'exp':  # exp value
                    scalarParamsStr += f'{scalarParam_name}=' \
                                       f'{scalar_param_val:.4E}|'
                elif scalarParam_opt['disp'] == 'int':  # integer value
                    scalarParamsStr += f'{scalarParam_name}=' \
                                       f'{scalar_param_val}|'
                else:  # float value
                    scalarParamsStr += f'{scalarParam_name}=' \
                                       f'{scalar_param_val:.4f}|'

        # get time series views
        viewsDict = {}
        for view in views:
            viewsDict[view] = np.array(example.features.feature[view].float_list.value)

        # plot features
        plot_features_example(viewsDict,
                              scalarParamsStr,
                              example_id_str,
                              example_label, plotDir, scheme,
                              basename=f'{basename}',
                              display=True)
