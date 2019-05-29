"""
Train models using the best configuration achieved on a hyperparameter optimization study.
"""

import tensorflow as tf
import os
import hpbandster.core.result as hpres

if 'nobackup' in os.path.dirname(__file__):
    from src.estimator_util import InputFn, ModelFn, CNN1dModel
    from src.eval_results import eval_model
    from src.config import Config
else:
    from src.estimator_util import InputFn, ModelFn, CNN1dModel
    from src.eval_results import eval_model
    from src.config import Config


def run_main(config):

    classifier = tf.estimator.Estimator(ModelFn(CNN1dModel, config),
                                        config=tf.estimator.RunConfig(keep_checkpoint_max=1,
                                                                      ),
                                        model_dir=config.model_dir_custom
                                        )

    train_input_fn = InputFn(file_pattern=config.tfrec_dir + '/train*', batch_size=config.batch_size,
                             mode=tf.estimator.ModeKeys.TRAIN, label_map=config.label_map, centr_flag=config.centr_flag)
    eval_input_fn = InputFn(file_pattern=config.tfrec_dir + '/val*', batch_size=config.batch_size,
                            mode=tf.estimator.ModeKeys.EVAL, label_map=config.label_map, centr_flag=config.centr_flag)

    result = []
    for epoch_i in range(1, config.n_epochs + 1):  # Train and evaluate the model for n_epochs
        print('\n\x1b[0;33;33m' + "Starting epoch %d of %d" % (epoch_i, config.n_epochs) + '\x1b[0m\n')
        _ = classifier.train(train_input_fn)

        print('\n\x1b[0;33;33m' + "Evaluating" + '\x1b[0m\n')
        res_eval = classifier.evaluate(eval_input_fn)

        confm_info = {key: value for key, value in res_eval.items() if key.startswith('label_')}

        res_i = {'loss': float(res_eval['loss']),
                 'val acc': float(res_eval['accuracy']),
                 # 'train prec': res_train['precision'],
                 'val prec': float(res_eval['precision']),
                 'confmx': confm_info,
                 'epoch': epoch_i}
        if not config.multi_class:
            res_i['roc auc'] = res_eval['roc auc']

        result.append(res_i)
        tf.logging.info('After epoch: {:d}: val acc: {:.6f}, val prec: {:.6f}'.format(epoch_i, res_i['val acc'],
                                                                                      res_i['val prec']))

    # eval_model(config, classifier, result)


if __name__ == '__main__':

    n_models = 1  # number of models in the ensemble

    # get best configuration from the HPO study
    res = hpres.logged_results_to_HBS_result('/home/msaragoc/Kepler_planet_finder/configs/study_2')
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    # best_config = id2config[incumbent]['config']
    best_config = id2config[(13, 0, 7)]['config']

    # Shallue's best configuration
    shallues_best_config = {'num_loc_conv_blocks': 2, 'init_fc_neurons': 512, 'pool_size_loc': 7,
                            'init_conv_filters': 4, 'conv_ls_per_block': 2, 'dropout_rate': 0, 'decay_rate': 1e-4,
                            'kernel_stride': 1, 'pool_stride': 2, 'num_fc_layers': 4, 'batch_size': 64, 'lr': 1e-5,
                            'optimizer': 'Adam', 'kernel_size': 5, 'num_glob_conv_blocks': 5, 'pool_size_glob': 5}

    config = best_config

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    for item in range(n_models):
        print('Training model %i out of %i' % (item + 1, n_models))
        # run_main(Config(**best_config))
        run_main(Config(**config))

