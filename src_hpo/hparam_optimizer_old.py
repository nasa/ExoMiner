# 3rd party
import os
import pickle
# import time
# import argparse
# import logging
# import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

# import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
# from hpbandster.optimizers import BOHB

# logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.DEBUG)
# logging.propagate = False

if 'nobackup' in os.path.dirname(__file__):
    # For running on Pleiades without matplotlib fig() error
    plt.switch_backend('agg')

# local
# from src_hpo.worker_tf_locglob import TransitClassifier as TransitClassifier_tf

###################################


def check_run_id(run_id, shared_directory, worker=False):
    def _gen_run_id(run_id):
        if run_id[-1].isnumeric():
            return run_id[:-1] + str(int(run_id[-1]) + 1)
        else:
            return run_id + str(1)

    while os.path.isfile(os.path.join(shared_directory, 'configs_%s.json' % run_id)):
        run_id = _gen_run_id(run_id)

    if worker:
        if run_id[-1] == '1':
            run_id = run_id[:-1]
        elif run_id[-1].isnumeric():
            run_id = run_id[:-1] + str(int(run_id[-1]) - 1)

    print('run_id: ' + run_id)

    return run_id


def analyze_results(result, args, shared_dir, run_id):

    # save results
    with open(os.path.join(shared_dir, 'results_%s.pkl' % run_id), 'wb') as fh:
        pickle.dump(result, fh)

    # # load the example run from the log files
    # result = hpres.logged_results_to_HBS_result('example_5_run/')

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()
    try:
        inc_config = id2conf[inc_id]['config']
        print('Best found configuration:', inc_config)
    except KeyError:
        print('No best found configuration!')

    all_runs = result.get_all_runs()

    print('A total of %i unique configurations were sampled.' % len(id2conf.keys()))
    print('A total of %i runs were executed.' % len(all_runs))
    print('Total budget corresponds to %.1f full function evaluations.'
          % (sum([r.budget for r in all_runs]) / args.max_budget))
    print('Total budget corresponds to %.1f full function evaluations.'
          % (sum([r.budget for r in all_runs]) / args.max_budget))
    print('The run took  %.1f seconds to complete.'
          % (all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_loss = inc_run.loss

    print('It achieved accuracies of %f (validation) and %f (test).' % (inc_run.info['validation accuracy'], 0.00))

    figs = {}

    # Observed losses grouped by budget
    figs['loss'], _ = hpvis.losses_over_time(all_runs)

    # Number of concurrent runs
    figs['concurrent_n'], _ = hpvis.concurrent_runs_over_time(all_runs)

    # Number of finished runs
    figs['finished_n'], _ = hpvis.finished_runs_over_time(all_runs)

    # Spearman rank correlation coefficients of the losses between different budgets
    figs['Spearman'], _ = hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    figs['model_vs_rand'], _ = hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    for figname, fig in figs.items():
        fig.set_size_inches(10, 8)
        fig.savefig(os.path.join(shared_dir, figname + '.png'), bbox_inches='tight')

    if 'nobackup' not in shared_dir:
        plt.show()

        def realtime_learning_curves(runs):
            """
            example how to extract a different kind of learning curve.

            The x values are now the time the runs finished, not the budget anymore.
            We no longer plot the validation loss on the y axis, but now the test accuracy.

            This is just to show how to get different information into the interactive plot.

            """
            sr = sorted(runs, key=lambda r: r.budget)
            lc = list(filter(lambda t: not t[1] is None,
                             [(r.time_stamps['finished'], r.info['validation accuracy']) for r in sr]))
            return [lc, ]

        try:
            lcs = result.get_learning_curves(lc_extractor=realtime_learning_curves)
            hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))
        except TypeError as e:
            print('\nGot TypeError: ', e)


class json_result_logger(hpres.json_result_logger):
    def __init__(self, directory, run_id, overwrite=False):
        """
        convenience logger for 'semi-live-results'

        Logger that writes job results into two files (configs.json and results.json).
        Both files contain proper json objects in each line.

        This version opens and closes the files for each result.
        This might be very slow if individual runs are fast and the
        filesystem is rather slow (e.g. a NFS).

        Parameters
        ----------

        directory: string
            the directory where the two files 'configs.json' and
            'results.json' are stored
        overwrite: bool
            In case the files already exist, this flag controls the
            behavior:

                * True:   The existing files will be overwritten. Potential risk of deleting previous results
                * False:  A FileExistsError is raised and the files are not modified.
        """
        os.makedirs(directory, exist_ok=True)

        self.config_fn = os.path.join(directory, 'configs_%s.json' % run_id)
        self.results_fn = os.path.join(directory, 'results_%s.json' % run_id)

        try:
            with open(self.config_fn, 'x') as fh:
                pass
        except FileExistsError:
            if overwrite:
                with open(self.config_fn, 'w') as fh:
                    pass
            else:
                raise FileExistsError('The file %s already exists.' % self.config_fn)
        except:
            raise

        try:
            with open(self.results_fn, 'x') as fh:
                pass
        except FileExistsError:
            if overwrite:
                with open(self.results_fn, 'w') as fh:
                    pass
            else:
                raise FileExistsError('The file %s already exists.' % self.config_fn)

        except:
            raise

        self.config_ids = set()


# def _start_worker(worker_i, worker, args, shared_directory, run_id, host, ns_host, ns_port):
#     print('Starting worker %d ..' % worker_i)
#
#     time.sleep(2)  # short artificial delay to make sure the nameserver is already running
#     w = worker(args, worker_id_custom=worker_i, run_id=run_id, host=host, nameserver=ns_host, nameserver_port=ns_port)
#     w.load_nameserver_credentials(working_directory=shared_directory)
#     w.run(background=False)
#     exit(0)


def get_ce_weights(label_map, tfrec_dir):
    filenames = [tfrec_dir + '/' + file for file in os.listdir(tfrec_dir)
                 if not file.startswith('test')]

    label_vec = []
    for file in filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        # try:
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8")
            label_vec.append(label_map[label])
        # except tf.errors.DataLossError as e:
        #     pass  # 916415

    label_counts = [label_vec.count(category) for category in range(max(label_map.values()) + 1)]
    ce_weights = [max(label_counts) / count_i for count_i in label_counts]

    return ce_weights, 'global_view_centr' in example.features.feature


# def run_main(args):
#
#     port = 9090
#     run_id = 'transit_classifier'
#
#     if 'tess' in args.tfrec_dir:
#         args.label_map = {"PC": 1, "NTP": 0, "EB": 2, "BEB": 2} if args.multi_class else {"PC": 1, "NTP": 0, "EB": 0, "BEB": 0}
#     else:
#         args.label_map = {"PC": 1, "NTP": 0, "AFP": 2} if args.multi_class else {"PC": 1, "NTP": 0, "AFP": 0}
#
#     # # Reduce dataset size; for prototyping
#     # fraction = 10
#     # for io_id in ['x', 'y']:
#     #     for set_id in ['train', 'val', 'test']:
#     #         data_dct[io_id][set_id] = data_dct[io_id][set_id][:int(len(data_dct[io_id][set_id])/fraction)]
#
#     shared_directory = os.path.dirname(__file__)
#
#     result_logger = json_result_logger(directory=shared_directory, run_id=run_id, overwrite=True)
#
#     # Start nameserver:
#     host = hpns.nic_name_to_host('lo')
#     name_server = hpns.NameServer(run_id=run_id, host=host, port=port, working_directory=shared_directory)
#     ns_host, ns_port = name_server.start()
#
#     worker = TransitClassifier_tf
#
#     # Define and run optimizer
#     bohb = BOHB(
#         configspace=worker.get_configspace(),
#         run_id=run_id,
#         host=host,
#         nameserver=ns_host,
#         nameserver_port=ns_port,
#         result_logger=result_logger,
#         min_budget=args.min_budget,
#         max_budget=args.max_budget,
#         # min_points_in_model=18,
#         eta=args.eta,
#         top_n_percent=15, num_samples=64, random_fraction=1 / 3,
#         bandwidth_factor=3, min_bandwidth=1e-3
#     )
#
#     args.ce_weights, args.centr_flag = get_ce_weights(args.label_map, tfrec_dir)
#
#     arguments = [(i, worker, args, shared_directory, run_id, host, ns_host, ns_port)
#                  for i in range(args.n_processes)]
#
#     pool = multiprocessing.Pool(processes=args.n_processes)
#     _ = [pool.apply_async(_start_worker, argument) for argument in arguments]
#     pool.close()
#
#     res = bohb.run(n_iterations=args.n_iterations)
#
#     # shutdown optimizer and nameserver
#     bohb.shutdown(shutdown_workers=True)
#     name_server.shutdown()
#
#     # Analyse and save results
#     analyze_results(res, args, shared_directory, run_id)
#
#
# if __name__ == '__main__':
#
#     min_budget = 4  # more info: hpbandster/optimizers/config_generators/bohb.py l.313
#     max_budget = 64
#     n_iterations = 16
#
#     n_processes = 4
#     eta = 2  # Down sampling rate
#
#
#     tfrec_dir = '/nobackupp2/plopesge/dr_24'
#
#
#     parser = argparse.ArgumentParser(description='Transit classifier hyperparameter optimizer')
#     parser.add_argument('--multi_class', type=bool, default=True)
#     parser.add_argument('--test_frac', type=float, default=0.1, help='data fraction for model testing')
#     parser.add_argument('--val_frac', type=float, default=0.1, help='model validation data fraction')
#     parser.add_argument('--tfrec_dir', type=str, default=tfrec_dir)
#
#     parser.add_argument('--min_budget', type=float, help='Minimum number of epochs for training.', default=min_budget)
#     parser.add_argument('--max_budget', type=float, help='Maximum number of epochs for training.', default=max_budget)
#     parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer',
#                         default=n_iterations)
#     parser.add_argument('--n_processes', type=int, help='Number of processes to run in parallel.', default=n_processes)
#     parser.add_argument('--eta', type=int, help='Down sampling rate', default=eta)
#     args = parser.parse_args()
#
#     run_main(args)
