"""
Run inference with a model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model, custom_object_scope
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import pandas as pd
import argparse
import yaml
from pathlib import Path
import logging
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import shap

# local
from models.models_keras import ExoMinerJointLocalFlux
from src.utils.utils_dataio import InputFnv2 as InputFn, set_tf_data_type_for_features
from models.models_keras import Time2Vec, SplitLayer
from src.utils.utils_dataio import get_data_from_tfrecords_for_predictions_table

def predict_model(config, model_path, res_dir, logger=None):

    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    # get data from TFRecords files to be displayed in the table with predictions
    data = get_data_from_tfrecords_for_predictions_table(config['datasets'],
                                                         config['data_fields'],
                                                         config['datasets_fps'])

    # load models
    if logger is None:
        print('Loading model...')
    else:
        logger.info('Loading model...')
    custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
    with custom_object_scope(custom_objects):
        old_model = load_model(filepath=model_path, compile=False)
        weights = old_model.get_weights()

    model_obj = ExoMinerJointLocalFlux(config, config['features_set'])
    model = model_obj.kerasModel
    model.set_weights(weights)

    if config['write_model_summary']:
        with open(res_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot model and save the figure
    if config['plot_model']:
        plot_model(model,
                   to_file=res_dir / 'model.png',
                   show_shapes=False,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=96)

    # get prediction and attention scores
    scores = {dataset: [] for dataset in config['datasets']}
    attn_scores_all = {dataset: [] for dataset in config['datasets']}
    for dataset in scores:

        if logger is None:
            print(f'Predicting on dataset {dataset}...')
        else:
            logger.info(f'Predicting on dataset {dataset}...')

        predict_input_fn = InputFn(
            file_paths=config['datasets_fps'][dataset],
            batch_size=config['inference']['batch_size'],
            mode='PREDICT',
            label_map=config['label_map'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
        )

        scores[dataset], attn_scores_all[dataset] = model.predict(
            predict_input_fn(),
            batch_size=None,
            verbose=config['verbose_model'],
            steps=None,
            callbacks=None,
        )

    branch_names_attn = model_obj.branch_names # branch order for attention model

    # Standardise attention tensor to (B, Q, K)
    attn3d = attn_scores_all['test']       
    np.save(res_dir / 'attn3d.npy', attn3d)       

    predictions = scores['test']    

    # SHAP analysis
    print("Running SHAP GradientExplainer…")

    # Create reference batch for gradient explainer
    MIN_REF = 32
    first_batch = next(iter(predict_input_fn()))
    feats_dict  = first_batch[0] if isinstance(first_batch, tuple) else first_batch
    batch_n     = next(iter(feats_dict.values())).shape[0]

    bg_idx = tf.constant(
        np.random.choice(batch_n, min(MIN_REF, batch_n), replace=False), dtype=tf.int32
    )
    background = {k: tf.gather(v, bg_idx, axis=0) for k, v in feats_dict.items()}
    input_order       = [t.name.split(":")[0] for t in old_model.inputs]
    background_inputs = [background[k].numpy() for k in input_order]

    explainer = shap.GradientExplainer(old_model, background_inputs)

    # Map tensors to branches (56 tensors, 11 branches)
    cfg = config['config']
    # collect branch names
    branch_names_shap = (
        list(cfg['conv_branches'].keys())     # 8 conv
        + list(cfg['scalar_branches'].keys())   # 2 scalar
        + ['diff_img']                          # 1 diff-image
    )                                
    tensor2branch = {}
    def register(names, br):
        for n in names:
            tensor2branch[n] = br

    for br, spec in cfg['conv_branches'].items():
        register(spec['views'],   br)
        if spec['scalars'] is not None:
            register(spec['scalars'], br)

    for br, names in cfg['scalar_branches'].items():
        register(names, br)

    register(cfg['diff_img_branch']['imgs'],         'diff_img')
    register(cfg['diff_img_branch']['imgs_scalars'], 'diff_img')
    register(cfg['diff_img_branch']['scalars'],      'diff_img')
    tensor_names = input_order

    # SHAP loop: aggregate to (batch, 11)
    shap_branch_batches = []

    for feats_batch in predict_input_fn():
        feats = feats_batch[0] if isinstance(feats_batch, tuple) else feats_batch
        input_list = [feats[k].numpy() for k in input_order]

        shap_list = explainer.shap_values(input_list)[0]  
        # bucket per branch
        agg = {br: [] for br in branch_names_shap}
        for t_name, sb in zip(tensor_names, shap_list):
            br = tensor2branch[t_name]
            sb = np.abs(sb)
            while sb.ndim > 2:          # collapse feature axes
                sb = sb.mean(axis=-1)
            agg[br].append(sb.mean(axis=1)) 

        # mean over tensors of the same branch
        per_branch = np.stack(
            [np.mean(agg[br], axis=0) for br in branch_names_shap], axis=1
        )                                # (batch, 11)
        shap_branch_batches.append(per_branch)

    shap_values_branch = np.concatenate(shap_branch_batches, axis=0)
    np.save(res_dir / "shap_branch_values.npy", shap_values_branch)
    print("SHAP done ->", shap_values_branch.shape)  # (N, 11)

    if predictions.ndim == 2:
        predictions = predictions.ravel()

    plot_branchwise_shap_correlation(shap_values_branch, predictions, branch_names_shap, res_dir,
                                     title="Correlation between SHAP and Prediction")

    scatter_shap_vs_prediction(shap_values_branch, predictions, branch_names_shap, res_dir)

    plot_mean_shap(shap_values_branch, branch_names_shap, res_dir)

    shap_vals_agg = np.abs(shap_values_branch)       # keep as-is
    shap_primary_idx = shap_vals_agg.argmax(axis=1)  # (300,)
    shap_primary_branch = [branch_names_shap[i] for i in shap_primary_idx]

    # Compute per-sample primary branch total attention RECEIVED by each branch = sum over queries
    attn_per_branch = attn3d.sum(axis=1)        
    primary_idx     = attn_per_branch.argmax(axis=1)    
    primary_branch  = [branch_names_attn[i] for i in primary_idx]

    print("\nSanity-check: summed attention per branch for sample 0")
    for k, br in enumerate(branch_names_attn):
        print(f"{k:2d}  {br:22s}  {attn_per_branch[0, k]:.4f}")
    print(f"→ primary_branch[0] = {primary_branch[0]}\n")

    plot_attention_vs_prediction(
        attn3d, predictions, branch_names_attn, res_dir,
        title="Correlation between Attention and Prediction"
    )

    plot_avg_attention_heatmap(
        attn3d, branch_names_attn, res_dir,
        title="Average Attention Across Dataset"
    )

    scatter_attention_vs_prediction(
        attn3d, predictions, branch_names_attn, res_dir
    )

    # Add scores & primary branch to data dict
    for dataset in config['datasets']:
        print(dataset)
        if not config['config']['multi_class']:
            data[dataset]['score'] = scores[dataset].ravel()
        else:
            for lbl, idx in config['label_map'].items():
                data[dataset][f'score_{lbl}'] = scores[dataset][:, idx]

        data[dataset]['attention_primary_branch'] = primary_branch
        data[dataset]['shap_primary_branch'] = shap_primary_branch

    # write results to a csv file
    for dataset in config['datasets']:

        data_df = pd.DataFrame(data[dataset])

        # sort in descending order of output
        # if not config['config']['multi_class']:
        #     data_df.sort_values(by='score', ascending=False, inplace=True)
        data_df.to_csv(res_dir / f'ranked_predictions_{dataset}set.csv', index=False)

def safe_linregress(x, y):
    if np.allclose(x, x[0]):
        return None
    slope, intercept, r, *_ = linregress(x, y)
    return slope, intercept, r

def plot_attention_vs_prediction(attn_scores,
                                 predictions,
                                 branch_names,
                                 res_dir,
                                 title="Attention vs Prediction Score"):
    """
    Correlate the total attention each branch receives with the model score.

    Parameters
    ----------
    attn_scores : np.ndarray
        Shape (B, Q, K)  *or*  (B, H, Q, K).
    predictions : np.ndarray
        Shape (B,)  or  (B, 1).
    branch_names : list[str]
        Ordered names of the K branches (tokens).
    """

    # normalize input (if necessary)
    if attn_scores.ndim == 4:              # (B, H, Q, K) -> (B, Q, K)
        attn_scores = attn_scores.mean(axis=1)
    assert attn_scores.ndim == 3, \
        "Expected (batch, query_len, key_len) after head-averaging"

    # collapse query dimension: sum attention over queries
    # attn_scores.sum(axis=1) -> (B, K)
    attention_to_each_branch = attn_scores.sum(axis=1)

    # correlation
    correlations = [
        np.corrcoef(attention_to_each_branch[:, k], predictions)[0, 1]
        for k in range(attention_to_each_branch.shape[1])
    ]

    # plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.array(correlations)[None, :],
        xticklabels=branch_names,
        yticklabels=["corr"],
        cmap="coolwarm",
        center=0,
        annot=True,
    )
    plt.title(title)
    plt.xlabel("Branch")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(res_dir / "attn_vs_pred.png"); plt.clf()


def plot_avg_attention_heatmap(attn_scores, branch_names, res_dir, title="Average Attention Across Dataset"):
    """
    Plots a heatmap of average attention scores across all batches.

    Parameters:
        attn_scores (np.ndarray): shape (batch_size, num_heads, query_len, key_len)
        branch_names (list of str): ordered branch names corresponding to token positions
        title (str): plot title
    """

    # Average across batch and heads
    avg_attn = attn_scores.mean(axis=0)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, xticklabels=branch_names, yticklabels=branch_names, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel("Key Branch")
    plt.ylabel("Query Branch")
    plt.tight_layout()
    plt.savefig(res_dir / "attn_avg_heatmap.png"); plt.clf()

def scatter_attention_vs_prediction(attn_scores,
                                    predictions,
                                    branch_names,
                                    res_dir: Path,
                                    add_trend=True,
                                    dpi=150,
                                    prefix="scatter_"):
    """
    Save a scatter plot for every branch: total attention (x) vs model prediction (y).

    Parameters
    ----------
    attn_scores : np.ndarray
        (B, Q, K) or (B, H, Q, K)
    predictions : np.ndarray
        (B,) or (B, 1)
    branch_names : list[str]
        Length K: ordered branch identifiers.
    res_dir : pathlib.Path | str
        Existing output directory (same one you use for the heat-map).
    add_trend : bool
        Overlay least-squares regression line and Pearson r if True.
    dpi : int
        Resolution of PNGs.
    prefix : str
        Filename prefix for each image (result = f"{prefix}{branch}.png").
    """
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # normalize if necessary
    if attn_scores.ndim == 4:              # (B, H, Q, K) → (B, Q, K)
        attn_scores = attn_scores.mean(axis=1)
    assert attn_scores.ndim == 3, "Need (batch, query_len, key_len)"

    # collapse query
    attention_to_each_branch = attn_scores.sum(axis=1)        # (B, K)

    # scatter per branch
    for k, br in enumerate(branch_names):
        x = attention_to_each_branch[:, k]
        y = predictions

        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, s=12)
        plt.xlabel("Total attention to branch")
        plt.ylabel("Prediction score")
        plt.title(f"{br}: attention vs prediction")

        if add_trend and len(x) > 1:
            res = safe_linregress(x, y)
        if res is not None:
            slope, intercept, r, *_ = res
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, slope*xs + intercept, linewidth=1)
            plt.text(0.02, 0.95, f"r = {r:.2f}",
                    ha="left", va="top", transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(res_dir / f"{prefix}{br}.png", dpi=dpi)
        plt.close()

    print(f"Saved {len(branch_names)} scatter plots to {res_dir.resolve()}")

def plot_branchwise_shap_correlation(shap_vals, predictions, branch_names, res_dir, title="SHAP vs Prediction Correlation"):
    """
    Save a correlation heatmap for every branch: SHAP (x) vs model prediction (y).

    Parameters
    ----------
    shap_vals : np.ndarray 
        (B, K)
    predictions : np.ndarray
        (B,) or (B, 1)
    branch_names : list[str]
        Length K: ordered branch identifiers.
    res_dir : pathlib.Path | str
        Existing output directory (same one you use for the heat-map).
    title: str
        Title of the plot.
    """
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim > 2:
        shap_vals = shap_vals.sum(axis=1)  # sum tokens if necessary

    correlations = [
        np.corrcoef(shap_vals[:, k], predictions)[0, 1]
        for k in range(shap_vals.shape[1])
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.array(correlations)[None, :],
        xticklabels=branch_names,
        yticklabels=["corr"],
        cmap="coolwarm",
        center=0,
        annot=True,
    )
    plt.title(title)
    plt.xlabel("Branch")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(res_dir / "shap_vs_pred.png")
    plt.clf()

def scatter_shap_vs_prediction(shap_vals, predictions, branch_names, res_dir: Path, add_trend=True, dpi=150, prefix="shap_scatter_"):
    """
    Save a scatter plot for every branch: SHAP (x) vs model prediction (y).

    Parameters
    ----------
    shap_vals : np.ndarray
        (B, K)
    predictions : np.ndarray
        (B,) or (B, 1)
    branch_names : list[str]
        Length K: ordered branch identifiers.
    res_dir : pathlib.Path | str
        Existing output directory (same one you use for the heat-map).
    add_trend : bool
        Overlay least-squares regression line and Pearson r if True.
    dpi : int
        Resolution of PNGs.
    prefix : str
        Filename prefix for each image (result = f"{prefix}{branch}.png").
    """
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim > 2:
        shap_vals = shap_vals.sum(axis=1)  # aggregate over tokens

    for k, br in enumerate(branch_names):
        x = shap_vals[:, k]
        y = predictions

        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, s=12)
        plt.xlabel("SHAP value (branch)")
        plt.ylabel("Prediction score")
        plt.title(f"{br}: SHAP vs prediction")

        if add_trend and len(x) > 1:
            slope, intercept, r, *_ = linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, slope * xs + intercept, linewidth=1)
            plt.text(0.02, 0.95, f"r = {r:.2f}", ha="left", va="top", transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(res_dir / f"{prefix}{br}.png", dpi=dpi)
        plt.close()

    print(f"Saved SHAP scatter plots to {res_dir.resolve()}")

def plot_mean_shap(shap_vals, branch_names, res_dir):
    mean_imp = np.abs(shap_vals).mean(axis=0)
    plt.figure(figsize=(8,4))
    sns.barplot(x=branch_names, y=mean_imp)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean |SHAP|")
    plt.title("Average Branch Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(res_dir / "shap_bar_mean.png"); plt.clf()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--model_config_fp', type=str, help='File path to YAML model configuration file.', default=None)
    parser.add_argument('--model_fp', type=str, help='Model file path.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    model_fp = Path(args.model_fp)
    config_fp = Path(args.config_fp)
    model_config_fp = Path(args.model_config_fp)
    output_dir = Path(args.output_dir)

    with(open(args.config_fp, 'r')) as file:
        predict_config = yaml.unsafe_load(file)

    with open(args.model_config_fp, 'r') as f:
        model_config = yaml.safe_load(f)

    predict_config['config'] = model_config

    # set up logger
    predict_config['logger'] = logging.getLogger(name=f'predict_model')
    logger_handler = logging.FileHandler(filename=output_dir / 'predict_model.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    predict_config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    predict_config['logger'].addHandler(logger_handler)
    predict_config['logger'].info(f'Starting evaluating model {model_fp} in {output_dir}')

    predict_model(predict_config, model_fp, output_dir, logger=predict_config['logger'])
