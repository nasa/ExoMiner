import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorboard.backend.event_processing import event_accumulator

if 'nobackup' in os.path.dirname(__file__):
    from estimator_util import InputFn, picklesave

    # For running on Pleiades without matplotlib fig() error
    plt.switch_backend('agg')
else:
    from src.estimator_util import InputFn, picklesave


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
            print('Elapsed: %s' % (time.time() - self.tstart))


def read_tfevents(outputFolder):
    inputLogFile = outputFolder + '/' + [file for file in os.listdir(outputFolder) if file.startswith('events')][0]

    with Timer():
        ea = event_accumulator.EventAccumulator(inputLogFile, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()  # loads events from file
        tags = ea.Tags()

    features = ['wall_time', 'step', 'value']
    out_dict = {}
    with Timer():
        for tag in tags['scalars']:
            if len(ea.Scalars(tag)) == 0:
                out_dict[tag] = None
            else:
                out_dict[tag] = {}
                for prop in features:
                    out_dict[tag][prop] = []
                    for entry in ea.Scalars(tag):
                        out_dict[tag][prop].append(getattr(entry, prop))

    return out_dict


def get_test_metric(config, classifier):
    test_input_fn = InputFn(file_pattern=config.tfrec_dir + '/test*', batch_size=config.batch_size,
                            mode=tf.estimator.ModeKeys.EVAL, label_map=config.label_map, centr_flag=config.centr_flag)

    print('\n\x1b[0;33;33m' + "Evaluating on test set" + '\x1b[0m\n')
    tf.logging.set_verbosity(tf.logging.INFO)

    test_res = classifier.evaluate(test_input_fn)

    confm_info = {key: value for key, value in test_res.items() if key.startswith('label_')}

    res = {'loss': float(test_res['loss']),
           'test acc': float(test_res['accuracy']),
           'test prec': float(test_res['precision']),
           'confmx': confm_info}

    if not config.multi_class:
        res['roc auc'] = test_res['roc auc']

    return res


def eval_train_history(config, val_result):
    train_history = read_tfevents(config.model_dir_custom)

    epoch_list = [i['epoch'] for i in val_result]
    figs = {}
    figname_map = {'accuracy': {'val': 'val acc', 'train': 'accuracy_1'},
                   'loss': {'val': 'loss', 'train': 'loss'},
                   'precision': {'val': 'val prec', 'train': 'precision_1'}}

    for figure_name in figname_map:
        val_str, train_str = figname_map[figure_name]['val'], figname_map[figure_name]['train']
        val_list = [i[val_str] for i in val_result]

        figs[figure_name] = plt.figure()
        ax1 = figs[figure_name].add_subplot(1, 1, 1)
        ax1.plot(epoch_list, val_list, 'C2')
        ax1.set_title('model ' + figure_name)
        ax1.set_ylabel('value [-]')
        ax1.set_xlabel('epoch')
        ax1.set_xlim(left=0, right=epoch_list[-1])

        if train_history[train_str]:
            train_list = train_history[train_str]['value']
            epoch_list_train = [i / (config.n_train_examples / config.batch_size) for i in
                                train_history[train_str]['step']]
            ax1.plot(epoch_list_train, train_list, 'C0')
            ax1.legend(['val set', 'train set'])
        else:
            ax1.legend(['val set'])

    if 'Documents' in os.path.dirname(__file__):
        for figure_name in figname_map:
            if figure_name in figs:
                figs[figure_name].show()

    for fig in figs:
        figs[fig].savefig(os.path.join(config.model_dir_custom, fig + '.png'), bbox_inches='tight')

    return train_history


def eval_model(config, classifier, eval_res):
    # import pickle
    # with open(config.model_dir_custom + '/eval_result.pickle', 'rb+') as fp:
    #     eval_res = pickle.load(fp)

    # res_sorted = sorted(eval_res, key=lambda x: x['validation accuracy'])

    train_res = eval_train_history(config, eval_res)
    test_res = get_test_metric(config, classifier)

    picklesave(config.model_dir_custom + '/result.pickle', {'train': train_res, 'eval': eval_res, 'test': test_res})

    print('Test set metrics:\n' + 'acc: {:.6f}, prec: {:.6f}'.format(test_res['test acc'], test_res['test prec']))
    print(', '.join([key + ': ' + '%d' % int(value) for key, value in test_res['confmx'].items()]))

    with open(config.model_dir_custom + '/result_test_set.txt', 'r') as fh:
        for key, value in [(i, val) for (i, val) in test_res.items() if i != 'confmx']:
            fh.write(key + ': {:.6f}'.format(value))
            fh.write('\n')

        if 'confmx' in test_res:
            fh.write('\n')
            for key, value in test_res['confmx'].items():
                fh.write(key + ': {:d}'.format(int(value)))
                fh.write('\n')
