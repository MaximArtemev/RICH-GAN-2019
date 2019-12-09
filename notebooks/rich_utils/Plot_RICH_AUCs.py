# coding: utf-8

import os

import numpy as np
import pandas as pd
import itertools
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as colors
from itertools import combinations as comb
from scipy import stats
import tensorflow as tf
import utils_rich_mrartemev as utils_rich
from sklearn.externals import joblib
from plot_utils import MidpointNormalize, plot_2d, score_func, \
    score_func_w, ks_2samp_w, interleave_ecdfs, get_ecdf,\
    merge_samples, rejection, weighted_mean, weighted_std,\
    integrate, get_metric, get_aucs_with_errors, calculate_metric_in_2d_bins

import argparse
import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('particle1', type=str,
                    help='First particle to analyze. proton, muon, electron, etc')
parser.add_argument('particle2', type=str,
                    help='Second particle to analyze. proton, muon, electron, etc')
parser.add_argument('particle1_predictor', type=str,
                    help='Name for particle1 predictor.'
                         ' Plz note that you should have file with the same name + "_preprocessor.pkl'
                         ' in the preprocessors folder')
parser.add_argument('particle2_predictor', type=str,
                    help='Name for particle2 predictor.'
                         ' Plz note that you should have file with the same name + "_preprocessor.pkl'
                         ' in the preprocessors folder')
parser.add_argument('--out_folder', type=str, default='Figures',
                    help='Directory where to save all your precious plots.'
                         ' For .gitignore sake name it like Figures_something_dir (starting with Figures)')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


PARTICLE1 = args.particle1
PARTICLE2 = args.particle2
PARTICLE1_PREDICTOR = args.particle1_predictor
PARTICLE2_PREDICTOR = args.particle2_predictor
FIGURES_FOLDER = args.out_folder


data_full_particle1 = utils_rich.load_and_merge_and_cut(utils_rich.datasets[PARTICLE1])
data_full_particle2 = utils_rich.load_and_merge_and_cut(utils_rich.datasets[PARTICLE2])

data_train_particle1, data_val_particle1, data_test_particle1 = utils_rich.split(data_full_particle1)
data_train_particle2, data_val_particle2, data_test_particle2 = utils_rich.split(data_full_particle2)

particle1_train = data_train_particle1.sample(data_test_particle1.shape[0])
particle2_train = data_train_particle2.sample(data_test_particle2.shape[0])

with tf.Session(config=tf_config) as sess:
    predictor_particle1 = tf.contrib.predictor.from_saved_model("exported_model/" + PARTICLE1_PREDICTOR)
    particle1_scaler = joblib.load(("preprocessors/" + PARTICLE1_PREDICTOR + "_preprocessor.pkl"))
    particle1_transformed = np.concatenate([
        particle1_scaler.transform(data_test_particle1.drop(utils_rich.weight_col, axis=1).values),
        data_test_particle1[utils_rich.weight_col].values[:, np.newaxis]
    ], axis=1)
    pred_dict_particle1 = predictor_particle1({"x": particle1_transformed[:, utils_rich.y_count:]})
    predicted_particle1_dlls = pred_dict_particle1["dlls"]

    gen_x_tr_particle1 = pred_dict_particle1.get("gen_x", None)

    # Stupid, but there seems to be no better out-of-the-box way
    particle1_transformed[:, :utils_rich.y_count] = predicted_particle1_dlls
    predicted_invtrans_particle1 = particle1_scaler.inverse_transform(particle1_transformed[:, :-1])[:,
                                   :utils_rich.y_count]
    gen_x_particle1 = particle1_scaler.inverse_transform(
        np.concatenate([predicted_particle1_dlls, gen_x_tr_particle1], axis=1)
    )[:, utils_rich.y_count:] if not (gen_x_tr_particle1 is None) else None

    for i, col in enumerate(utils_rich.dll_columns):
        data_test_particle1["predicted_{}".format(col)] = predicted_invtrans_particle1[:, i]

with tf.Session(config=tf_config) as sess:
    predictor_particle2 = tf.contrib.predictor.from_saved_model("exported_model/" + PARTICLE2_PREDICTOR)
    particle2_scaler = joblib.load(("preprocessors/" + PARTICLE2_PREDICTOR + "_preprocessor.pkl"))
    particle2_transformed = np.concatenate([
        particle2_scaler.transform(data_test_particle2.drop(utils_rich.weight_col, axis=1).values),
        data_test_particle2[utils_rich.weight_col].values[:, np.newaxis]
    ], axis=1)
    pred_dict_particle2 = predictor_particle2({"x": particle2_transformed[:, utils_rich.y_count:]})
    predicted_particle2_dlls = pred_dict_particle2["dlls"]

    gen_x_tr_particle2 = pred_dict_particle2.get("gen_x", None)

    # Stupid, but there seems to be no better out-of-the-box way
    particle2_transformed[:, :utils_rich.y_count] = predicted_particle2_dlls
    predicted_invtrans_particle2 = particle2_scaler.inverse_transform(particle2_transformed[:, :-1])[:,
                                   :utils_rich.y_count]
    gen_x_particle2 = particle2_scaler.inverse_transform(
        np.concatenate([predicted_particle2_dlls, gen_x_tr_particle2], axis=1)
    )[:, utils_rich.y_count:] if not (gen_x_tr_particle2 is None) else None

    for i, col in enumerate(utils_rich.dll_columns):
        data_test_particle2["predicted_{}".format(col)] = predicted_invtrans_particle2[:, i]

os.makedirs(FIGURES_FOLDER, exist_ok=True)

if not (gen_x_particle1 is None):
    fig, axx = plt.subplots(len(utils_rich.raw_feature_columns), 1,
                            figsize=(12, 8 * len(utils_rich.raw_feature_columns)))

    for i, col in enumerate(utils_rich.raw_feature_columns):
        _, bins, _ = axx[i].hist(data_test_particle1[col], weights=data_test_particle1[utils_rich.weight_col],
                                 label='Data', bins=100, alpha=0.5, normed=True)
        axx[i].hist(gen_x_particle1[:, i], label='Generated', bins=bins, alpha=0.5, normed=True)
        axx[i].legend()
        axx[i].set_title("{}, {}".format(PARTICLE1, col))
        if 'Brunel_P' == col:
            axx[i].set_yscale("log", nonposy='clip')
    fig.savefig("{}/gen_x_{}.pdf".format(FIGURES_FOLDER, PARTICLE1))
    plt.clf()

if not (gen_x_particle2 is None):
    fig, axx = plt.subplots(len(utils_rich.raw_feature_columns), 1,
                            figsize=(12, 8 * len(utils_rich.raw_feature_columns)))

    for i, col in enumerate(utils_rich.raw_feature_columns):
        _, bins, _ = axx[i].hist(data_test_particle2[col], weights=data_test_particle2[utils_rich.weight_col],
                                 label='Data', bins=100, alpha=0.5, normed=True)
        axx[i].hist(gen_x_particle2[:, i], label='Generated', bins=bins, alpha=0.5, normed=True)
        axx[i].legend()
        axx[i].set_title("{}, {}".format(PARTICLE2, col))
        if 'Brunel_P' == col:
            axx[i].set_yscale("log", nonposy='clip')
    fig.savefig("{}/gen_x_{}.pdf".format(FIGURES_FOLDER, PARTICLE2))
    plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
empty = set(axes)

ranges_of_interest_particle1 = {}
MIN_ENTRIES = 1000

for col, ax in zip(utils_rich.dll_columns, axes):
    real, gen = data_test_particle1[col].values, data_test_particle1["predicted_{}".format(col)].values
    l, r = min(real.min(), gen.min()), max(real.max(), gen.max())
    bins = np.linspace(l, r, 101)
    h1, _, _ = ax.hist(real, label='real', bins=bins, weights=data_test_particle1[utils_rich.weight_col], alpha=0.5)
    h2, _, _ = ax.hist(gen, label='generated', bins=bins, weights=data_test_particle1[utils_rich.weight_col], alpha=0.5)
    ax.set_yscale("log", nonposy='clip')
    ax.set_title("Protons, {}".format(col))  # name
    ax.legend()

    l = bins[min((h1 >= MIN_ENTRIES).argmax(), (h2 >= MIN_ENTRIES).argmax())]
    r = bins[::-1][min((h1[::-1] >= MIN_ENTRIES).argmax(), (h2[::-1] >= MIN_ENTRIES).argmax())]
    ranges_of_interest_particle1[col] = (l, r)
    empty.remove(ax)
for ax in empty: ax.axis('off')
fig.savefig("{}/DLLs_log_{}.pdf".format(FIGURES_FOLDER, PARTICLE1))
plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
empty = set(axes)

ranges_of_interest_particle2 = {}
MIN_ENTRIES = 1000

for col, ax in zip(utils_rich.dll_columns, axes):
    real, gen = data_test_particle2[col].values, data_test_particle2["predicted_{}".format(col)].values
    l, r = min(real.min(), gen.min()), max(real.max(), gen.max())
    bins = np.linspace(l, r, 101)
    h1, _, _ = ax.hist(real, label='real', bins=bins, weights=data_test_particle2[utils_rich.weight_col], alpha=0.5)
    h2, _, _ = ax.hist(gen, label='generated', bins=bins, weights=data_test_particle2[utils_rich.weight_col], alpha=0.5)
    ax.set_yscale("log", nonposy='clip')
    ax.set_title("Protons, {}".format(col))  # name
    ax.legend()

    l = bins[min((h1 >= MIN_ENTRIES).argmax(), (h2 >= MIN_ENTRIES).argmax())]
    r = bins[::-1][min((h1[::-1] >= MIN_ENTRIES).argmax(), (h2[::-1] >= MIN_ENTRIES).argmax())]
    ranges_of_interest_particle2[col] = (l, r)
    empty.remove(ax)
for ax in empty: ax.axis('off')
fig.savefig("{}/DLLs_log_{}.pdf".format(FIGURES_FOLDER, PARTICLE2))
plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
empty = set(axes)
for col, ax in zip(utils_rich.dll_columns, axes):
    real, gen = data_test_particle1[col].values, data_test_particle1["predicted_{}".format(col)].values
    bins = np.linspace(*ranges_of_interest_particle1[col], 101)
    ax.hist(real, label='real', bins=bins, weights=data_test_particle1[utils_rich.weight_col], alpha=0.5)
    ax.hist(gen, label='generated', bins=bins, weights=data_test_particle1[utils_rich.weight_col], alpha=0.5)
    ax.set_title("{}, {}".format(PARTICLE1, col))
    ax.legend()
    empty.remove(ax)
for ax in empty: ax.axis('off')

fig.savefig("{}/DLLs_linear_{}.pdf".format(FIGURES_FOLDER, PARTICLE1))
plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
empty = set(axes)
for col, ax in zip(utils_rich.dll_columns, axes):
    real, gen = data_test_particle2[col].values, data_test_particle2["predicted_{}".format(col)].values
    bins = np.linspace(*ranges_of_interest_particle2[col], 101)
    ax.hist(real, label='real', bins=bins, weights=data_test_particle2[utils_rich.weight_col], alpha=0.5)
    ax.hist(gen, label='generated', bins=bins, weights=data_test_particle2[utils_rich.weight_col], alpha=0.5)
    ax.set_title("{}, {}".format(PARTICLE2, col))
    ax.legend()
    empty.remove(ax)
for ax in empty: ax.axis('off')

fig.savefig("{}/DLLs_linear_{}.pdf".format(FIGURES_FOLDER, PARTICLE2))
plt.clf()

for col1, col2 in comb(utils_rich.dll_columns, 2):
    fig = plot_2d(data_val_particle1[col1].values,
                  data_val_particle1[col2].values,
                  data_val_particle1[utils_rich.weight_col].values,
                  data_test_particle1["predicted_{}".format(col1)].values,
                  data_test_particle1["predicted_{}".format(col2)].values,
                  data_test_particle1[utils_rich.weight_col].values,
                  col1, col2, PARTICLE1, bins=30)

    fig.savefig("{}/{}_vs_{}_{}.pdf".format(FIGURES_FOLDER, col1, col2, PARTICLE1))
    plt.clf()

for col1, col2 in comb(utils_rich.dll_columns, 2):
    fig = plot_2d(data_val_particle2[col1].values,
                  data_val_particle2[col2].values,
                  data_val_particle2[utils_rich.weight_col].values,
                  data_test_particle2["predicted_{}".format(col1)].values,
                  data_test_particle2["predicted_{}".format(col2)].values,
                  data_test_particle2[utils_rich.weight_col].values,
                  col1, col2, PARTICLE2, bins=30)

    fig.savefig("{}/{}_vs_{}_{}.pdf".format(FIGURES_FOLDER, col1, col2, PARTICLE2))
    plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
ax_empty = set(ax for ax in axes)
w = data_test_particle1[utils_rich.weight_col].values
for col, ax in zip(utils_rich.dll_columns, axes):
    real, gen = data_test_particle1[col].values, data_test_particle1["predicted_{}".format(col)].values
    xs, ys = score_func_w(real, w, data_val_particle1[col].values, data_val_particle1[utils_rich.weight_col].values)
    ax.plot(xs, ys - xs, label='Data')
    xs, ys = score_func_w(real, w, gen, w)
    ax.plot(xs, ys - xs, label='Generated')
    ax.set_xlabel("Data efficiency")
    ax.set_ylabel("(Test efficiency) - (Data efficiency)")
    ax.set_title("{}, {}".format(PARTICLE1, col))
    ax.legend()
    ax_empty.remove(ax)
plt.tight_layout()

for ax in ax_empty:
    ax.axis('off')

fig.savefig("{}/dEff_{}.pdf".format(FIGURES_FOLDER, PARTICLE1))
plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
ax_empty = set(ax for ax in axes)
w = data_test_particle2[utils_rich.weight_col].values
for col, ax in zip(utils_rich.dll_columns, axes):
    real, gen = data_test_particle2[col].values, data_test_particle2["predicted_{}".format(col)].values
    xs, ys = score_func_w(real, w, data_val_particle2[col].values, data_val_particle2[utils_rich.weight_col].values)
    ax.plot(xs, ys - xs, label='Data')
    xs, ys = score_func_w(real, w, gen, w)
    ax.plot(xs, ys - xs, label='Generated')
    ax.set_xlabel("Data efficiency")
    ax.set_ylabel("(Test efficiency) - (Data efficiency)")
    ax.set_title("{}, {}".format(PARTICLE2, col))
    ax.legend()
    ax_empty.remove(ax)
plt.tight_layout()

for ax in ax_empty:
    ax.axis('off')

fig.savefig("{}/dEff_{}.pdf".format(FIGURES_FOLDER, PARTICLE2))
plt.clf()
# I have no idea what this is doing. Probably should ask someone about it.

# print(ks_2samp_w(data_test_particle1.RichDLLk.values, data_test_particle1.probe_sWeight.values,
#                  data_test_particle1.predicted_RichDLLk.values, data_test_particle1.probe_sWeight.values))
#
# print(ks_2samp_w(data_test_particle1.RichDLLk.values, data_test_particle1.probe_sWeight.values,
#                  data_train_particle1.RichDLLk.values, data_train_particle1.probe_sWeight.values))


# I have no idea what this is doing. Probably should ask someone about it.

# p1, w1 = get_ecdf(data_test_particle1.RichDLLk.values, data_test_particle1.probe_sWeight.values)
# p2, w2 = get_ecdf(data_test_particle1.predicted_RichDLLk.values, data_test_particle1.probe_sWeight.values)

# plt.plot(p1[:300], w1[:300])
# plt.plot(p2[:300], w2[:300])

# Probably better to make it print something

# get_ipython().run_line_magic('timeit', 'x, y1, y2 = interleave_ecdfs(p1, w1, p2, w2)')

# x, y1, y2 = interleave_ecdfs(p1, w1, p2, w2)
#
# plt.plot(x, y1)
# plt.plot(x, y2)
#
# i = np.abs(y2 - y1).argmax()
# s = slice(i - 5, i + 5)
# plt.plot(x[s], y1[s])
# plt.plot(x[s], y2[s])
# plt.axhline(y=y1[i])
# plt.axhline(y=y2[i])


data_full_test = merge_samples(data_test_particle1, data_test_particle2)

_, bins, _ = plt.hist(data_test_particle1.RichDLLk, weights=data_test_particle1.probe_sWeight,
                      label='data_test_{}'.format(PARTICLE1), alpha=0.5, bins=300)

_, _, _ = plt.hist(data_test_particle2.RichDLLk, weights=data_test_particle2.probe_sWeight,
                   label='data_test_{}'.format(PARTICLE2), alpha=0.5, bins=bins)
plt.legend()
plt.savefig("{}/RichDLLk_stuff.pdf".format(FIGURES_FOLDER))
plt.clf()

with open('{}/rejection_logs.txt'.format(FIGURES_FOLDER), 'w') as f:
    for x in np.linspace(0.01, 0.99, 99):
        f.write("{} - {}\n".format(x, rejection(x,
                                                data_test_particle1.RichDLLk.values,
                                                data_test_particle2.RichDLLk.values,
                                                data_test_particle1.probe_sWeight.values,
                                                data_test_particle2.probe_sWeight.values)))

with open('{}/score_func_w_logs.txt'.format(FIGURES_FOLDER), 'w') as f:
    for x, y in zip(*score_func_w(data_test_particle1.RichDLLk.values, data_test_particle1.probe_sWeight.values,
                                  data_test_particle2.RichDLLk.values, data_test_particle2.probe_sWeight.values)):
        f.write("{}, {}\n".format(x, y))

metrics = ['rej90', 'mean0', 'mean1', 'std0', 'std1', 'auc', 'auc_']
metrics_resolved = get_metric(data_full_test, metrics=metrics)
with open('{}/metrics.txt'.format(FIGURES_FOLDER), 'w') as f:
    for metric_, metric_result in zip(metrics, metrics_resolved):
        f.write("{} = {}, {}\n".format(metric_, metrics_resolved[0], metrics_resolved[1]))

fpr_test, tpr_test, _, = roc_curve(data_full_test.y_true,
                                   data_full_test.RichDLLk,
                                   sample_weight=data_full_test[utils_rich.weight_col])
tpr_test_, fpr_test_ = score_func_w(data_test_particle1.RichDLLk.values, data_test_particle1.probe_sWeight.values,
                                    data_test_particle2.RichDLLk.values, data_test_particle2.probe_sWeight.values, 500)
fpr_gen, tpr_gen, _, = roc_curve(data_full_test.y_true,
                                 data_full_test.predicted_RichDLLk,
                                 sample_weight=data_full_test[utils_rich.weight_col])

auc_test = roc_auc_score(data_full_test.y_true,
                         data_full_test.RichDLLk,
                         sample_weight=data_full_test[utils_rich.weight_col]
                         )
auc_gen = roc_auc_score(data_full_test.y_true,
                        data_full_test.predicted_RichDLLk,
                        sample_weight=data_full_test[utils_rich.weight_col]
                        )

N_SUBSAMPLES = 100

auc_test_, auc_test_e, auc_gen_, auc_gen_e = get_aucs_with_errors(data_full_test, N_SUBSAMPLES, )

print(r"""N subsamples: {} ({} for data and {} for gen)

        Full sample AUC (data): {:.5f}
     Subsample mean AUC (data): {:.5f}
     STD/sqrt(nsamples) (data): {:.5f}

   Full sample AUC (generated): {:.5f}
Subsample mean AUC (generated): {:.5f}
STD/sqrt(nsamples) (generated): {:.5f}
""".format(2 * N_SUBSAMPLES, N_SUBSAMPLES, N_SUBSAMPLES,
           auc_test,
           auc_test_,
           auc_test_e,
           auc_gen,
           auc_gen_,
           auc_gen_e,
           ))

with open('{}/auc_report.txt'.format(FIGURES_FOLDER), 'w') as f:
    f.write(r"""N subsamples: {} ({} for data and {} for gen)

        Full sample AUC (data): {:.5f}
     Subsample mean AUC (data): {:.5f}
     STD/sqrt(nsamples) (data): {:.5f}

   Full sample AUC (generated): {:.5f}
Subsample mean AUC (generated): {:.5f}
STD/sqrt(nsamples) (generated): {:.5f}
""".format(2 * N_SUBSAMPLES, N_SUBSAMPLES, N_SUBSAMPLES,
           auc_test,
           auc_test_,
           auc_test_e,
           auc_gen,
           auc_gen_,
           auc_gen_e,
           ))

fpr_test_ = 1. - fpr_test_
tpr_test_ = 1. - tpr_test_
plt.figure(figsize=(15, 15))
plt.plot(fpr_test[(tpr_test < 0.91) & (tpr_test > 0.89)], tpr_test[(tpr_test < 0.91) & (tpr_test > 0.89)], label='Data')
plt.plot(fpr_test_[(tpr_test_ < 0.91) & (tpr_test_ > 0.89)], tpr_test_[(tpr_test_ < 0.91) & (tpr_test_ > 0.89)],
         label='Data_')
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - {} vs {} - RichDLLk".format(data_test_particle1, data_test_particle2))

import seaborn as sns
sns.set()

metric = 'ks'
aucs_data, aucs_gen, aucs_diff, _, fig = calculate_metric_in_2d_bins(data_full_test,
                                                                     metric=metric,
                                                                     bins1=8,
                                                                     bins2=8,
                                                                     n_subsamples=10,
                                                                     central_only=True, min_bin_entries=20,
                                                                     plot_distributions=True)

fig, ((ax_data, ax_gen),
      (ax_diff, ax_empty)) = plt.subplots(2, 2, figsize=(16, 15))

ax_empty.axis('off')

sns.heatmap(aucs_data.T,
            annot=True,
            ax=ax_data,
            );
ax_data.set_title("KS p-value, {}".format(PARTICLE2));

sns.heatmap(aucs_gen.T,
            annot=True,
            ax=ax_gen,
            );
ax_gen.set_title("KS p-value, {}".format(PARTICLE1))

ax_diff.axis('off')
fig.tight_layout();
fig.savefig("{}/RichDLLk_dPdETA.pdf".format(FIGURES_FOLDER))
plt.clf()

np.set_printoptions(precision=6, linewidth=150)
np.save('{}/aucs_data'.format(FIGURES_FOLDER), aucs_data)

# Removed global weird_data from plot_utils
# roc_auc_score(weird_data.y_true, weird_data.RichDLLk, sample_weight=weird_data.probe_sWeight)

_, bins, _ = plt.hist(np.log10(data_full_particle1.Brunel_P), weights=data_full_particle1.probe_sWeight, bins=300,
                      label=PARTICLE1, alpha=0.5)
_, _, _ = plt.hist(np.log10(data_full_particle2.Brunel_P), weights=data_full_particle2.probe_sWeight, bins=bins,
                   label=PARTICLE2,
                   alpha=0.5)
plt.yscale('log')
plt.xlabel("log10(P)")
plt.legend()
plt.savefig('{}/data_full_weights_stuff_log10'.format(FIGURES_FOLDER))
plt.clf()

_, bins, _ = plt.hist((data_full_particle1.Brunel_ETA), weights=data_full_particle1.probe_sWeight, bins=300,
                      label=PARTICLE1, alpha=0.5)
_, _, _ = plt.hist((data_full_particle2.Brunel_ETA), weights=data_full_particle2.probe_sWeight, bins=bins,
                   label=PARTICLE2,
                   alpha=0.5)
plt.yscale('log')
plt.xlabel("Eta")
plt.legend()
plt.savefig('{}/data_full_weights_stuff_Brunel_ETA'.format(FIGURES_FOLDER))
plt.clf()
print('Finished')

# ToDo -> save this to one plot
#
# n_samples = 100
# batch_size = len(data_test_particle1) // n_samples
# n_bins = 30
# for col in utils_rich.dll_columns:
#     fig, axx = plt.subplots(1, 2, figsize=(18, 7))
#
#     _, bins = np.histogram(data_test_particle1["predicted_{}".format(col)], bins=n_bins,
#                            weights=data_test_particle1[utils_rich.weight_col])
#
#     hists = np.array([np.histogram(data_test_particle1["predicted_{}".format(col)].values[i:i + batch_size],
#                                    bins=bins,
#                                    weights=data_test_particle1[utils_rich.weight_col].values[i:i + batch_size])[0]
#                       for i in range(0, batch_size * n_samples, batch_size)])
#     pulls = (hists - hists.mean(axis=0)) / hists.std(axis=0)
#     bin_correlations = (pulls[:, np.newaxis, :] * pulls[:, :, np.newaxis]).mean(axis=0)
#     img = axx[0].imshow(bin_correlations, cmap='bwr', norm=MidpointNormalize(midpoint=0.))
#     axx[0].set_title('Generated')
#     axx[0].set_xlabel(col)
#     axx[0].set_ylabel(col)
#     fig.colorbar(img, ax=axx[0])
#
#     hists = np.array([np.histogram(data_test_particle1[col].values[i:i + batch_size],
#                                    bins=bins,
#                                    weights=data_test_particle1[utils_rich.weight_col].values[i:i + batch_size])[0]
#                       for i in range(0, batch_size * n_samples, batch_size)])
#     pulls = (hists - hists.mean(axis=0)) / hists.std(axis=0)
#     bin_correlations = (pulls[:, np.newaxis, :] * pulls[:, :, np.newaxis]).mean(axis=0)
#     img = axx[1].imshow(bin_correlations, cmap='bwr', norm=MidpointNormalize(midpoint=0.))
#     axx[1].set_title('Real')
#     axx[1].set_xlabel(col)
#     axx[1].set_ylabel(col)
#     fig.colorbar(img, ax=axx[1])
#

# <H1>THE END</H1>
#
# kaon_double_transformed = pd.DataFrame(
#     particle1_scaler.inverse_transform(particle1_scaler.transform(data_val_particle1.values)),
#     columns=data_val_particle1.columns)
# pion_double_transformed = pd.DataFrame(pion_scaler.inverse_transform(pion_scaler.transform(pion_train_dt.values)),
#                                        columns=pion_train_dt.columns)
#
# # In[15]:
#
#
# data_pi_k = pd.concat([data_test_particle1, data_test_particle2], axis=0, ignore_index=True)
# data_pi_k["is_kaon"] = np.concatenate([np.ones(len(data_test_particle1)), np.zeros(len(data_test_particle2))])
#
# # In[16]:
#
#
# data_pi_k_train = pd.concat([particle1_train, particle2_train], axis=0, ignore_index=True)
# data_pi_k_train["is_kaon"] = np.concatenate([np.ones(len(particle1_train)), np.zeros(len(particle2_train))])
#
# # In[17]:
#
#
# data_pi_k_double_transformed = pd.concat([kaon_double_transformed, pion_double_transformed], axis=0, ignore_index=True)
# data_pi_k_double_transformed["is_kaon"] = np.concatenate(
#     [np.ones(len(kaon_double_transformed)), np.zeros(len(pion_double_transformed))])
#
# # In[18]:
#
#
# data_pi_k_double_transformed.shape
#
# # In[19]:
#
#
# data_pi_k.shape
#
#
# # In[20]:
#
#
# def compute_auc_by_bins(true_labels, predictions, dataset, binning_columns, bins_count, bin_examples):
#     assert len(binning_columns) == 2
#     edges = [None, None]
#     counts, edges[0], edges[1] = np.histogram2d(dataset[binning_columns[0]],
#                                                 dataset[binning_columns[1]], bins_count)
#     aucs = np.ndarray(shape=counts.shape)
#     aucs[:, :] = np.nan
#     for index_tuple in itertools.product(range(counts.shape[0]), range(counts.shape[1])):
#         if counts[index_tuple] < bin_examples:
#             continue
#         cut_datasets_indices = utils.select_by_cuts(dataset,
#                                                     {binning_columns[0]: edges[0][index_tuple[0]:index_tuple[0] + 2],
#                                                      binning_columns[1]: edges[1][
#                                                                          index_tuple[1]:index_tuple[1] + 2]}).index
#
#         aucs[index_tuple] = roc_auc_score(true_labels[cut_datasets_indices], predictions[cut_datasets_indices])
#     return aucs, edges
#
#
# # In[21]:
#
#
# aucs_gen, edges = compute_auc_by_bins(data_pi_k.is_kaon, data_pi_k.predicted_dll_kaon, data_pi_k,
#                                       ["particle_one_eta", "particle_two_eta"], 10, int(1e3))
#
# # In[22]:
#
#
# aucs_data, edges_data = compute_auc_by_bins(data_pi_k.is_kaon, data_pi_k.dll_kaon, data_pi_k,
#                                             ["particle_one_eta", "particle_two_eta"], 10, int(1e3))
#
# # In[23]:
#
#
# aucs_train, edges_train = compute_auc_by_bins(data_pi_k_train.is_kaon, data_pi_k_train.dll_kaon, data_pi_k_train,
#                                               ["particle_one_eta", "particle_two_eta"], 10, int(1e3))
#
# # In[24]:
#
#
# aucs_double_transformed, edges_double_transformed = compute_auc_by_bins(
#     data_pi_k_double_transformed.is_kaon, data_pi_k_double_transformed.dll_kaon, data_pi_k_double_transformed,
#     ["particle_one_eta", "particle_two_eta"], 10, int(1e3))
#
# # In[25]:
#
#
# # assert all(map(np.array_equal, edges, edges_data))
# # assert all(map(np.array_equal, edges, edges_train))
#
#
# # In[26]:
#
#
# roc_auc_score(data_pi_k.is_kaon, data_pi_k.dll_kaon)
#
# # In[27]:
#
#
# roc_auc_score(data_pi_k.is_kaon, data_pi_k.predicted_dll_kaon)
#
# # In[28]:
#
#
# roc_auc_score(data_pi_k_train.is_kaon, data_pi_k_train.dll_kaon)
#
# # In[29]:
#
#
# roc_auc_score(data_pi_k_double_transformed.is_kaon, data_pi_k_double_transformed.dll_kaon)
#
# # In[30]:
#
#
# import seaborn as sns
#
#
# # In[31]:
#
#
# def get_bins_labels(edges):
#     return list(map("{:.1f}".format, 0.5 * (edges[1:] + edges[:-1])))
#
#
# def get_bins(edges):
#     return 0.5 * (edges[1:] + edges[:-1])
#
#
# x_bins_labels = get_bins_labels(edges[0])
# y_bins_labels = get_bins_labels(edges[1])
# x_bins = get_bins(edges[0])
# y_bins = get_bins(edges[1])
#
# fig, ((ax_data, ax_diff_train),
#       (ax_diff_pred, ax_diff_doble_transformed)) = plt.subplots(2, 2, figsize=(16, 15))
#
# sns.heatmap(aucs_data.T, xticklabels=x_bins_labels, yticklabels=y_bins_labels, annot=True, ax=ax_data,
#             vmin=0.5, vmax=1);
# ax_data.invert_yaxis();
# ax_data.set_ylabel("$\eta_2$");
# ax_data.set_title("$\pi$ vs K, AUC(test data)");
#
# sns.heatmap((aucs_data - aucs_train).T, xticklabels=x_bins_labels, yticklabels=y_bins_labels, annot=True,
#             ax=ax_diff_train)
# ax_diff_train.invert_yaxis()
# ax_diff_train.set_title("$\pi$ vs K, AUC(test data)- AUC(train)");
#
# sns.heatmap((aucs_data - aucs_gen).T, xticklabels=x_bins_labels, yticklabels=y_bins_labels, annot=True, ax=ax_diff_pred)
# ax_diff_pred.invert_yaxis()
# ax_diff_pred.set_title("$\pi$ vs K, AUC(test data)- AUC(predicted)");
# ax_diff_pred.set_xlabel("$\eta_1$");
# ax_diff_pred.set_ylabel("$\eta_2$");
#
# sns.heatmap((aucs_data - aucs_double_transformed).T, xticklabels=x_bins_labels, yticklabels=y_bins_labels, annot=True,
#             ax=ax_diff_doble_transformed)
# ax_diff_doble_transformed.invert_yaxis()
# ax_diff_doble_transformed.set_title("$\pi$ vs K, AUC(test data)- AUC(double transformed test)");
# ax_diff_doble_transformed.set_xlabel("$\eta_1$");
# fig.tight_layout();
# fig.savefig("AUCs_%s.pdf" % MODEL_CORE)
#
# # In[32]:
#
#
# fig, ax = plt.subplots()
# ax.hist(data_test_particle2.dll_kaon, bins=100, density=True)
# ax.hist(data_test_particle1.dll_kaon, alpha=0.5, bins=100, density=True);
# ax.set_yscale("log")
#
# # In[33]:
#
#
# fig, ax = plt.subplots()
# ax.hist(data_test_particle2.predicted_dll_kaon, bins=100, density=True)
# ax.hist(data_test_particle1.predicted_dll_kaon, alpha=0.5, bins=100, density=True);
# ax.set_yscale("log")
#
# # In[34]:
#
#
# fig, ax = plt.subplots()
# ax.hist(data_test_particle1.dll_kaon, bins=100, density=True)
# ax.hist(data_test_particle1.predicted_dll_kaon, alpha=0.5, bins=100, density=True);
# ax.set_yscale("log")
#
# # In[35]:
#
#
# kaons_raw = data_test[data_test.particle_one_type == 3].drop(columns="particle_one_type")
#
# # In[36]:
#
#
# reverse_dlls = particle1_scaler.inverse_transform(particle1_scaler.transform(kaons_raw))[:, 1]
#
# # In[37]:
#
#
# fig, ax = plt.subplots()
# ax.hist(kaons_raw.dll_kaon, bins=100, density=True)
# ax.hist(reverse_dlls, alpha=0.5, bins=100, density=True);
# ax.set_yscale("log")
