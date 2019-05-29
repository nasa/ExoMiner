import os
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def evalplots_ensemble(eval_dir, models_eval, metrics, datasets):

    color_codes = {'val': 'r', 'test': 'k'}
    displ = 'curves'  # 'band'
    alpha = 0.3

    for m in metrics:
        f, ax = plt.subplots()
        for d in datasets:
            mvalues = []
            for model_eval in models_eval:
                mvalue = np.load(model_eval).item()[d][m]
                mvalues.append(mvalue)

                if displ == 'curves':
                    ax.plot(mvalue, alpha=alpha, color=color_codes[d])

            avg = np.mean(mvalues, axis=0)

            ax.plot(avg, color=color_codes[d], label=m)

            if displ == 'band':
                std = np.std(mvalues, axis=0, ddof=1)
                ax.plot(avg + std, linestyle='--', color=color_codes[d])
                ax.plot(avg - std, linestyle='--', color=color_codes[d])
                # ax.fill_between(np.arange(len(mvalue)), avg - std, avg + std, color=color_codes[d], alpha=alpha)

            ax.set_ylabel('Value')
            ax.set_xlabel('Epochs')
            ax.set_xlim([0.0, len(mvalue) + 1])
            if m != 'loss':
                ax.set_ylim([0, 1])
            ax.legend(loc="lower right")
            ax.set_title('{}'.format(m))
        f.savefig(eval_dir + '/ensemble_plots_{}.png'.format(m))


if __name__ == '__main__':

    eval_dir = '/home6/msaragoc/work_dir/HPO_Kepler_TESS/train_results/run_study_8/'
    models_eval = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if os.path.isfile(os.path.join(eval_dir, f))
                   and 'res_eval.npy' in f]
    metrics = ['precision', 'recall', 'pr auc', 'roc auc', 'accuracy', 'loss']
    datasets = ['val', 'test']

    evalplots_ensemble(eval_dir, models_eval, metrics, datasets)
