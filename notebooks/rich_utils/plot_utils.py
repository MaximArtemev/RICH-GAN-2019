import os

import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import combinations as comb
from scipy import stats
from scipy.stats.distributions import kstwobign

import tensorflow as tf
import utils_rich_mrartemev as utils_rich
from sklearn.externals import joblib


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_2d(col1_real, col2_real, w_real,
            col1_gen, col2_gen, w_gen,
            col1_title, col2_title, label, print_stats=True, bins=30):
    hist_real, bins_x, bins_y = np.histogram2d(col1_real, col2_real, weights=w_real, bins=bins)
    hist_gen, _, _ = np.histogram2d(col1_gen, col2_gen, weights=w_gen, bins=(bins_x, bins_y))
    s_real2, _, _ = np.histogram2d(col1_real, col2_real, weights=w_real ** 2, bins=(bins_x, bins_y))
    s_gen2, _, _ = np.histogram2d(col1_gen, col2_gen, weights=w_gen ** 2, bins=(bins_x, bins_y))
    s_total = (s_real2 + s_gen2) ** 0.5

    SAFE_CHI2_THR = 5
    valid_bins = (hist_real + hist_gen >= SAFE_CHI2_THR)
    n_excl_bins = ((~valid_bins) & ((hist_real > 0) | (hist_gen > 0))).sum()
    chi2 = (((hist_gen - hist_real) ** 2)[valid_bins] / (s_real2 + s_gen2)[valid_bins]).sum()
    ndf = valid_bins.sum() - 1
    p_value = 1 - stats.chi2.cdf(chi2, ndf)
    summary = r"""Good bins (sum>{thr}): {ngood}
Bad bins (too low stats, but not 0): {nbad}
chi2: {chi2}
ndf: {ndf}
p_value: {pval}
""".format(thr=SAFE_CHI2_THR, ngood=valid_bins.sum(), nbad=n_excl_bins, chi2=chi2, ndf=ndf, pval=p_value)

    fig, ((ax_real, ax_gen),
          (ax_compare, ax_compare2),
          (ax_compare3, ax_pull)) = plt.subplots(3, 2, figsize=(16, 21))

    img_real = ax_real.imshow(hist_real.T, origin='lower', cmap='Blues', extent=(bins_x[0], bins_x[-1],
                                                                                 bins_y[0], bins_y[-1]),
                              aspect='auto')
    fig.colorbar(img_real, ax=ax_real)
    ax_real.set_xlabel(col1_title)
    ax_real.set_ylabel(col2_title)
    ax_real.set_title('{}, Real'.format(label))

    img_gen = ax_gen.imshow(hist_gen.T, origin='lower', cmap='Blues', extent=(bins_x[0], bins_x[-1],
                                                                              bins_y[0], bins_y[-1]),
                            aspect='auto')
    fig.colorbar(img_gen, ax=ax_gen)
    ax_gen.set_xlabel(col1_title)
    ax_gen.set_ylabel(col2_title)
    ax_gen.set_title('{}, Generated'.format(label))

    ratio = hist_gen / np.where(hist_real == 0, 1.E+10, hist_real)
    img_ratio = ax_compare.imshow(ratio.T, origin='lower', cmap='seismic', extent=(bins_x[0], bins_x[-1],
                                                                                   bins_y[0], bins_y[-1]),
                                  aspect='auto', norm=MidpointNormalize(midpoint=1.))
    fig.colorbar(img_ratio, ax=ax_compare)
    ax_compare.set_xlabel(col1_title)
    ax_compare.set_ylabel(col2_title)
    ax_compare.set_title('{}, Generated / Real (0 -> 1.E+10)'.format(label))

    ratio = hist_gen / np.where(hist_real == 0, 1.E-10, hist_real)
    img_ratio = ax_compare2.imshow(ratio.T, origin='lower', cmap='seismic', extent=(bins_x[0], bins_x[-1],
                                                                                    bins_y[0], bins_y[-1]),
                                   aspect='auto', norm=MidpointNormalize(midpoint=1.))
    fig.colorbar(img_ratio, ax=ax_compare2)
    ax_compare2.set_xlabel(col1_title)
    ax_compare2.set_ylabel(col2_title)
    ax_compare2.set_title('{}, Generated / Real (0 -> 1.E-10)'.format(label))

    compare3 = (hist_gen - hist_real) / np.where(s_total == 0, 1.E+99, s_total)
    img_diff = ax_compare3.imshow(compare3.T, origin='lower', cmap='bwr', extent=(bins_x[0], bins_x[-1],
                                                                                  bins_y[0], bins_y[-1]),
                                  aspect='auto', norm=MidpointNormalize(midpoint=0.))
    fig.colorbar(img_diff, ax=ax_compare3)
    ax_compare3.set_xlabel(col1_title)
    ax_compare3.set_ylabel(col2_title)
    ax_compare3.set_title("{}, (gen - real) / sqrt(sigma_gen^2 + sigma_real^2)".format(label))

    mask = (hist_gen != 0) | (hist_real != 0)
    print("mean: {}; std: {}".format(compare3[mask].mean(), compare3[mask].std()))
    ax_pull.hist(compare3[mask], bins=100)
    ax_pull.set_title("pull")
    ax_pull.set_xlabel("{}, (gen - real) / sqrt(sigma_gen^2 + sigma_real^2)".format(label))

    if print_stats:
        ax_pull.annotate(summary, xy=(0.02, 0.75), xycoords='axes fraction')

    return fig


def score_func(col1, col2, n_points=100):
    assert len(col1.shape) == 1
    assert len(col2.shape) == 1

    col1_s = np.sort(col1)
    quantiles = np.linspace(0, col1.shape[0], n_points, endpoint=False).astype(np.int)
    thresholds = col1_s[quantiles].reshape([1, -1])
    ys = (col2.reshape([-1, 1]) < thresholds).sum(axis=0).astype(np.float) / col2.shape[0]
    xs = quantiles.astype(np.float) / col1.shape[0]

    return xs, ys


def score_func_w(col1, w1, col2, w2, n_points=100):
    assert len(col1.shape) == 1
    assert len(col2.shape) == 1
    assert len(w1.shape) == 1
    assert len(w2.shape) == 1

    i1 = col1.argsort()
    col1, w1 = col1[i1], w1[i1]
    cum_w1 = np.cumsum(w1)

    assert cum_w1[-1] > 0
    steps = np.linspace(0, cum_w1[-1], n_points, endpoint=False)
    thresholds = col1[np.argmax(cum_w1[:, np.newaxis] >= steps[np.newaxis, :], axis=0)]

    assert (thresholds[1:] - thresholds[:-1] >= 0).all(), "THIS SENTENCE IS WRONG"

    i2 = col2.argsort()
    col2, w2 = col2[i2], w2[i2]
    cum_w2 = np.cumsum(w2)

    assert cum_w2[-1] > 0
    ys = cum_w2[np.argmin(col2[:, np.newaxis] < thresholds[np.newaxis, :], axis=0)] / cum_w2[-1]
    xs = steps / cum_w1[-1]
    return xs, ys


def get_ecdf(points, weights=None):
    if weights is None:
        weights = np.ones_like(points).astype(float)

    assert len(points) == len(weights)

    i = np.argsort(points)
    p, w = points[i], weights[i]

    w_cumsum = np.cumsum(w)

    assert w_cumsum[-1] > 0

    w_cumsum /= w_cumsum[-1]

    return p, w_cumsum


def interleave_ecdfs(x1, y1, x2, y2):
    assert x1.shape == y1.shape
    assert x2.shape == y2.shape

    buf_x = np.concatenate([x1, x2])
    buf_y1 = np.concatenate([y1, np.empty(shape=(len(y2),))])
    buf_y2 = np.concatenate([np.empty(shape=(len(y1),)), y2])
    mask2 = np.concatenate([np.zeros_like(x1), np.ones_like(x2)]).astype(bool)
    mask1 = ~mask2
    i = np.argsort(buf_x)

    buf_x = buf_x[i]
    buf_y1 = buf_y1[i]
    buf_y2 = buf_y2[i]
    mask1 = mask1[i]
    mask2 = mask2[i]

    ileft1 = np.argmax(mask1)
    ileft2 = np.argmax(mask2)

    buf_y1[:ileft1] = 0
    buf_y2[:ileft2] = 0
    mask1[:ileft1] = True
    mask2[:ileft2] = True

    while not mask1.all():
        buf_y1[1:][~mask1[1:]] = buf_y1[:-1][~mask1[1:]]
        mask1[1:][~mask1[1:]] = mask1[:-1][~mask1[1:]]

    while not mask2.all():
        buf_y2[1:][~mask2[1:]] = buf_y2[:-1][~mask2[1:]]
        mask2[1:][~mask2[1:]] = mask2[:-1][~mask2[1:]]

    return buf_x, buf_y1, buf_y2


def ks_2samp_w(data1, w1, data2, w2):
    _, y1, y2 = interleave_ecdfs(*get_ecdf(data1, w1),
                                 *get_ecdf(data2, w2))
    d = np.abs(y2 - y1).max()
    en = (w1.sum() * w2.sum() / (w1.sum() + w2.sum())) ** 0.5
    prob = kstwobign.sf(en * d)
    prob = kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    return d, prob


def merge_samples(kaon_sample, pion_sample):
    y_true_kaons = np.ones(shape=(len(kaon_sample),), dtype=bool)
    y_true_pions = np.zeros(shape=(len(pion_sample),), dtype=bool)
    y_true = np.concatenate([y_true_kaons, y_true_pions], axis=0)

    result = pd.concat([kaon_sample, pion_sample], ignore_index=True)
    result['y_true'] = y_true
    return result


def rejection(eff, y_sig, y_bg, w_sig, w_bg):
    y_sig, y_bg = -y_sig, -y_bg
    i = np.argsort(y_sig)
    y_sig = y_sig[i]
    w_sig = w_sig[i]
    wsum = np.cumsum(w_sig)
    imin = np.argmax(wsum / wsum[-1] > eff)
    imax = len(wsum) - np.argmax(wsum[::-1] / wsum[-1] <= eff)

    assert imax >= imin, "{} {}".format(imin, imax)
    imax = max(imin + 1, imax)

    thr = (y_sig[imin:imax] * w_sig[imin:imax]).sum() / w_sig[imin:imax].sum()

    return w_bg[y_bg >= thr].sum() / w_bg.sum()


def weighted_mean(x, w):
    return (x * w).sum() / w.sum()


def weighted_std(x, w):
    return weighted_mean((x - weighted_mean(x, w)) ** 2, w) ** 0.5


def integrate(x, y):
    dx = x[1:] - x[:-1]
    y_ = 0.5 * (y[:-1] + y[1:])
    return (y_ * dx).sum()


def get_metric(
        ds,
        metrics,
        weight_col=utils_rich.weight_col,
        data_col='RichDLLk',
        gen_col='predicted_RichDLLk',
        truth_col='y_true',
):
    result = []
    ds0 = ds.loc[ds[truth_col] == 0]
    ds1 = ds.loc[ds[truth_col] == 1]

    w = [ds0[weight_col].values,
         ds1[weight_col].values]
    y_data = [ds0[data_col].values,
              ds1[data_col].values]
    y_gen = [ds0[gen_col].values,
             ds1[gen_col].values]

    for m in metrics:
        if m[:3] == 'rej':
            eff = float(m[3:]) / 100.
            assert (0. < eff) and (eff < 1.)

            result.append((rejection(eff, y_data[1], y_data[0], w[1], w[0]),
                           rejection(eff, y_gen[1], y_gen[0], w[1], w[0])))

        elif m[:4] == 'mean':
            part = int(m[4])
            result.append((weighted_mean(y_data[part], w[part]),
                           weighted_mean(y_gen[part], w[part])))
        elif m[:3] == 'std':
            part = int(m[3])
            result.append((weighted_std(y_data[part], w[part]),
                           weighted_std(y_gen[part], w[part])))
        elif m == 'auc':
            tpr_data, fpr_data = score_func_w(y_data[1], w[1],
                                              y_data[0], w[0], 500)
            tpr_gen, fpr_gen = score_func_w(y_gen[1], w[1],
                                            y_gen[0], w[0], 500)
            tpr_data = np.concatenate([[0.], tpr_data, [1.]])
            fpr_data = np.concatenate([[0.], fpr_data, [1.]])
            tpr_gen = np.concatenate([[0.], tpr_gen, [1.]])
            fpr_gen = np.concatenate([[0.], fpr_gen, [1.]])

            fpr_data, tpr_data = 1. - fpr_data[::-1], 1. - tpr_data[::-1]
            fpr_gen, tpr_gen = 1. - fpr_gen[::-1], 1. - tpr_gen[::-1]

            result.append((integrate(fpr_data, tpr_data),
                           integrate(fpr_gen, tpr_gen)))
        elif m == 'auc_':
            tpr_data, fpr_data = score_func_w(-y_data[1], w[1],
                                              -y_data[0], w[0], 500)
            tpr_gen, fpr_gen = score_func_w(-y_gen[1], w[1],
                                            -y_gen[0], w[0], 500)

            tpr_data = np.concatenate([[0.], tpr_data, [1.]])
            fpr_data = np.concatenate([[0.], fpr_data, [1.]])
            tpr_gen = np.concatenate([[0.], tpr_gen, [1.]])
            fpr_gen = np.concatenate([[0.], fpr_gen, [1.]])

            result.append((integrate(fpr_data, tpr_data),
                           integrate(fpr_gen, tpr_gen)))
        elif m == 'ks':
            _, p0 = ks_2samp_w(y_data[0], w[0],
                               y_gen[0], w[0])
            _, p1 = ks_2samp_w(y_data[1], w[1],
                               y_gen[1], w[1])
            result.append((p0, p1))
        else:
            raise NotImplementedError()

    return result


def get_aucs_with_errors(
        ds,
        n_subsamples=100,
        weight_col=utils_rich.weight_col,
        data_col='RichDLLk',
        gen_col='predicted_RichDLLk',
        truth_col='y_true',
        max_subsample_len_variation=0.01
):
    """
    Calculate data and gen AUCs with errors.
    As data and gen AUCs might be correlated, the sample is split into
    2 * n_subsamples parts, first n_subsamples to estimate the data AUC
    and last n_subsamples to estimate the generator AUC.
    """

    shuffle = np.arange(len(ds))
    np.random.shuffle(shuffle)
    splits = np.round(np.linspace(0, len(ds), 2 * n_subsamples + 1, endpoint=True)).astype(int)
    slices = [shuffle[l:r] for l, r in zip(splits[:-1], splits[1:])]
    assert len(slices) == 2 * n_subsamples

    smallest_len = len(min(slices, key=len))
    biggest_len = len(max(slices, key=len))

    if (biggest_len - smallest_len > max_subsample_len_variation * smallest_len):
        print("Warning: sample variation is too large. Min: {}, Max: {}, thr: {}".format(
            smallest_len, biggest_len,
            max_subsample_len_variation))

    aucs_data = [roc_auc_score(ds.iloc[s][truth_col],
                               ds.iloc[s][data_col],
                               sample_weight=ds.iloc[s][weight_col] if weight_col is not None else None)
                 for s in slices[:n_subsamples]]
    aucs_gen = [roc_auc_score(ds.iloc[s][truth_col],
                              ds.iloc[s][gen_col],
                              sample_weight=ds.iloc[s][weight_col] if weight_col is not None else None)
                for s in slices[n_subsamples:]]

    return np.mean(aucs_data), np.std(aucs_data) / n_subsamples ** 0.5, np.mean(aucs_gen), np.std(
        aucs_gen) / n_subsamples ** 0.5


# In[17]:


def get_aucs(
        ds,
        weight_col=utils_rich.weight_col,
        data_col='RichDLLk',
        gen_col='predicted_RichDLLk',
        truth_col='y_true'
):
    """
    Calculate data and gen AUCs.
    """
    auc_data = roc_auc_score(ds[truth_col],
                             ds[data_col],
                             sample_weight=ds[weight_col] if weight_col is not None else None)
    auc_gen = roc_auc_score(ds[truth_col],
                            ds[gen_col],
                            sample_weight=ds[weight_col] if weight_col is not None else None)

    # Removed global weird_data from plot_utils
    #    if auc_data >= 1: weird_data = ds

    return auc_data, auc_gen


def calculate_metric_in_2d_bins(
        ds,
        metric,
        bins1=5,
        bins2=6,
        var1='Brunel_P',
        var2='Brunel_ETA',
        log10_var1=True,
        weight_col=utils_rich.weight_col,
        disc_data='RichDLLk',
        disc_gen='predicted_RichDLLk',
        disc_truth='y_true',
        max_subsample_len_variation=0.3,
        n_subsamples=30,
        min_bin_entries=600,
        central_only=False,
        plot_distributions=False,
        fig_scale=6,
):
    col1 = np.log10(ds[var1].values) if log10_var1 else ds[var1].values
    col2 = ds[var2].values
    w = ds[weight_col].values

    _, bins1, bins2 = np.histogram2d(col1, col2, weights=w, bins=(bins1, bins2))
    hist0, _, _ = np.histogram2d(col1[ds[disc_truth].values == 0],
                                 col2[ds[disc_truth].values == 0],
                                 weights=w[ds[disc_truth].values == 0], bins=(bins1, bins2))
    hist1, _, _ = np.histogram2d(col1[ds[disc_truth].values == 1],
                                 col2[ds[disc_truth].values == 1],
                                 weights=w[ds[disc_truth].values == 1], bins=(bins1, bins2))

    gen_bins = lambda b: ((l, r) for l, r in zip(b[:-1], b[1:]))

    shape = (len(bins1) - 1, len(bins2) - 1)
    aucs_data = np.empty(shape);
    aucs_data[:] = np.nan
    aucs_gen = np.empty(shape);
    aucs_gen[:] = np.nan
    aucs_data_error = np.empty(shape);
    aucs_data_error[:] = np.nan
    aucs_gen_error = np.empty(shape);
    aucs_gen_error[:] = np.nan

    if plot_distributions:
        fig, axx = plt.subplots(len(bins2) - 1, len(bins1) - 1,
                                figsize=(fig_scale * len(bins1) - fig_scale,
                                         fig_scale * len(bins2) - fig_scale))
        axx_empty = set(axx.flatten())
    for i1, b1 in enumerate(gen_bins(bins1)):
        for i2, b2 in enumerate(gen_bins(bins2)):
            if hist0[i1, i2] >= min_bin_entries and hist1[i1, i2] >= min_bin_entries:
                selection = ((col1 >= b1[0]) &
                             (col1 < b1[1]) &
                             (col2 >= b2[0]) &
                             (col2 < b2[1]))
                if plot_distributions:
                    axx_empty.remove(axx[i2, i1])
                    ds_1 = ds.loc[selection & (ds[disc_truth] == 1)]
                    ds_0 = ds.loc[selection & (ds[disc_truth] == 0)]
                    _, dll_bins, _ = axx[i2, i1].hist(ds_1[disc_data],
                                                      weights=ds_1[weight_col],
                                                      bins=int(np.ceil(len(ds_1) ** 0.5)), label='1 data', alpha=0.5)
                    _, _, _ = axx[i2, i1].hist(ds_0[disc_data],
                                               weights=ds_0[weight_col],
                                               bins=dll_bins, label='0 data', alpha=0.5)
                    _, _, _ = axx[i2, i1].hist(ds_1[disc_gen],
                                               weights=ds_1[weight_col],
                                               bins=dll_bins, label='1 gen', alpha=0.6,
                                               histtype='step', linewidth=2.)
                    _, _, _ = axx[i2, i1].hist(ds_0[disc_gen],
                                               weights=ds_0[weight_col],
                                               bins=dll_bins, label='0 gen', alpha=0.6,
                                               histtype='step', linewidth=2.)
                    axx[i2, i1].legend(fontsize=18)
                    axx[i2, i1].set_xlabel(disc_data)

                    label_1 = "{:.2} <= {} < {:.2}".format(
                        b1[0],
                        "log10({})".format(var1) if log10_var1 else var1,
                        b1[1]
                    )
                    axx[i2, i1].text(0.5, 1.0, label_1,
                                     bbox={'facecolor': 'white', 'pad': 3},
                                     transform=axx[i2, i1].transAxes,
                                     horizontalalignment='center',
                                     fontsize=18)

                    label_2 = "{:.2} > {} >= {:.2}".format(b2[1], var2, b2[0])
                    axx[i2, i1].text(0.0, 0.5, label_2,
                                     bbox={'facecolor': 'white', 'pad': 3},
                                     transform=axx[i2, i1].transAxes,
                                     verticalalignment='center',
                                     rotation='vertical',
                                     fontsize=18)
                try:
                    if central_only:
                        # aucs_data[i1, i2], \
                        # aucs_gen [i1, i2]  = get_aucs(
                        #                         ds.loc[selection],
                        #                         weight_col=weight_col,
                        #                         data_col=disc_data,
                        #                         gen_col=disc_gen,
                        #                         truth_col=disc_truth,
                        #                     )
                        aucs_data[i1, i2], aucs_gen[i1, i2] = get_metric(
                            ds.loc[selection],
                            metrics=[metric],
                            weight_col=weight_col,
                            data_col=disc_data,
                            gen_col=disc_gen,
                            truth_col=disc_truth,
                        )[0]
                    else:
                        raise NotImplementedError()

                except Exception as e:
                    print("Something bad in bin {} {}".format(i1, i2))
                    print(e)
                    raise

    aucs_diff = aucs_gen - aucs_data
    aucs_diff_error = (aucs_gen_error ** 2 + aucs_data_error ** 2) ** 0.5

    if plot_distributions:
        for ax in axx_empty:
            ax.axis('off')
        fig.tight_layout()
    # fig.show()

    return [aucs_data, aucs_gen, aucs_diff, aucs_diff_error] + ([fig] if plot_distributions else [])
