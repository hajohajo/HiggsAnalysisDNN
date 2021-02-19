import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout' : True})
import seaborn as sns
sns.set()
sns.set_style("whitegrid")

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

from matplotlib import font_manager
font_manager._rebuild()
rcParams['font.serif'] = "Helvetica"
rcParams['font.family'] = "serif"

BKG_COLOR = sns.xkcd_rgb['cerulean']
SIG_COLOR = sns.xkcd_rgb['rouge']

LOSS_COLOR = sns.xkcd_rgb['dark sky blue']
DISCO_COLOR = sns.xkcd_rgb['dark rose']
AUC_COLOR = sns.xkcd_rgb['emerald']
ColorDict = { "sig":SIG_COLOR,
             "bkg":BKG_COLOR,
             "loss":LOSS_COLOR,
             "disco":DISCO_COLOR,
             "auc":AUC_COLOR}

EventClassColors = {
    0: SIG_COLOR,
    1: sns.xkcd_rgb['emerald green'],
    2: sns.xkcd_rgb['orange'],
    3: sns.xkcd_rgb['purple'],
    4: sns.xkcd_rgb['blue grey'],
    5: sns.xkcd_rgb['bright red'],
    6: sns.xkcd_rgb['teal']

}


from Datasets import EventTypeDict, InvertedEventTypeDict, BinningDict
class Plotter():
    def __init__(self, dataset, scaled_dataset, history):
        self._dataset = dataset
        self._scaled_dataset = scaled_dataset
        self._history = history

    def output_distribution(self):
        fig, (ax_0, ax_1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]},
                                         figsize=(6, 8))
        is_signal = self._dataset.loc[:, "event_id"] == 0
        sig_pred = self._dataset.loc[is_signal, "predicted_values"]
        bkg_pred = self._dataset.loc[~is_signal, "predicted_values"]
        sig_mt = self._dataset.loc[is_signal, "TransverseMass"]
        bkg_mt = self._dataset.loc[~is_signal, "TransverseMass"]
        binning = np.linspace(0.0, 1.0, 11)
        bin_centers = binning[:-1] + (binning[1:]-binning[:-1])/2.0

        sig_bin_means = -np.ones(len(bin_centers))
        sig_bin_std = np.zeros(len(bin_centers))
        bkg_bin_means = -np.ones(len(bin_centers))
        bkg_bin_std = np.zeros(len(bin_centers))

        sig_indices = np.digitize(sig_pred, binning)
        sig_indices[sig_indices==0] = 1
        sig_indices[sig_indices==len(binning)] = len(binning)-1
        sig_indices = sig_indices - 1
        bkg_indices = np.digitize(bkg_pred, binning)
        bkg_indices[bkg_indices==0] = 1
        bkg_indices[bkg_indices==len(binning)] = len(binning)-1
        bkg_indices = bkg_indices - 1

        for i in np.unique(sig_indices):
            sig_bin_means[i], sig_bin_std[i] = norm.fit(sig_mt[sig_indices == i])

        for i in np.unique(bkg_indices):
            bkg_bin_means[i], bkg_bin_std[i] = norm.fit(bkg_mt[bkg_indices == i])

        sig_up = np.add(sig_bin_means, sig_bin_std/2.0)
        sig_down = np.add(sig_bin_means, -sig_bin_std/2.0)
        bkg_up = np.add(bkg_bin_means, bkg_bin_std/2.0)
        bkg_down = np.add(bkg_bin_means, -bkg_bin_std/2.0)


        ax_0.hist((bkg_pred, sig_pred), bins=binning, label=["background", "signal"], color=[BKG_COLOR, SIG_COLOR],
                  stacked=True, edgecolor='k', linewidth=1.5, alpha=0.7, density=True)
        ax_0.legend()
        ax_0.set_xlim(0.0, 1.0)
        ax_1.set_xlabel("DNN prediction")
        ax_1.set_ylabel("Mean m$_{T}$")
        ax_0.set_ylabel("Normalized events/bin width")

        self._plot_class(bkg_bin_means, bkg_up, bkg_down, bin_centers, binning, ax_1, BKG_COLOR, "background")
        self._plot_class(sig_bin_means, sig_up, sig_down, bin_centers, binning, ax_1, SIG_COLOR, "signal")


        plt.savefig("plots/output_distr.pdf")
        plt.clf()

    def distortions(self):
        working_points = [0.1, 0.4, 0.65]
        line_types = ['dashed', 'dashdot', 'dotted']
        mt_bins = np.linspace(0.0, 500.0, 21)
        centers = mt_bins[:-1] + (mt_bins[1]-mt_bins[0])/2.0
        classes = InvertedEventTypeDict.keys()
        for _class in classes:
            if _class == 0:
                indices = self._dataset.loc[:, "event_id"] != 0
            else:
                indices = self._dataset.loc[:, "event_id"] == _class
            fig, (ax_0, ax_1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]},
                                             figsize=(6, 8))

            dnn_outputs = self._dataset.loc[indices, "predicted_values"]
            weights = self._dataset.loc[indices, "event_weight"]
            mt_before = self._dataset.loc[indices, "TransverseMass"]
            bin_cont_before, _, _ = ax_0.hist(mt_before, bins=mt_bins, label="No cut", histtype='step', linewidth=2.0, color=EventClassColors[_class])

            for i, working_point in enumerate(working_points):
                mt_after = mt_before.loc[dnn_outputs >= working_point]
                bin_cont_after, bins_, _ = ax_0.hist(mt_after, bins=mt_bins, linestyle=line_types[i], linewidth=2.0, label="Cut: "+str(working_point), histtype='step', color=EventClassColors[_class])
                ratio = np.ones_like(bin_cont_before)
                ratio[bin_cont_before != 0] = bin_cont_after[bin_cont_before != 0]/bin_cont_before[bin_cont_before != 0]
                ax_1.plot(np.append(np.insert(centers, 0, -mt_bins[1]), 2*mt_bins[-1]), np.append(np.insert(ratio, 0, ratio[0]), ratio[-1]), marker="o", linestyle=line_types[i], c='k')

            fig.suptitle("Sample distribution w.r.t. m$_{T}$")
            ax_0.legend()
            ax_0.set_ylabel("Events/bin")
            ax_0.set_xlim(-2.0, mt_bins[-1])
            ax_0.set_yscale('log')
            ax_1.set_xlabel("m$_{T}$ (GeV)")
            ax_1.set_ylabel("Ratio")
            ax_1.set_ylim(0.001, 1.05)
            plt.savefig("plots/"+InvertedEventTypeDict[_class]+"_distortion.pdf")
            plt.clf()


    def metrics(self, metrics_to_plot, labels, plot_val=True):
        x_binning = range(1, len(self._history.history['loss'])+1)
        x_ticks = np.array(x_binning)[np.array(x_binning) % 20 == 0]
        fig, ax = plt.subplots(nrows=1, ncols=1,
                                         figsize=(8, 6))
        for i, metric_label in enumerate(metrics_to_plot):
            metric = self._history.history[metric_label]
            color = ColorDict[metric_label]
            plt.plot(x_binning, metric, label=labels[i], c=color, linewidth=2.0)
            if (plot_val):
                val_metric = self._history.history["val_"+metric_label]
                plt.plot(x_binning, val_metric, label="val. "+labels[i], ls="--", c=color, linewidth=2.0)

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metric value")
        ax.set_xticks(x_ticks)
        ax.set_ylim(0.0, 1.25)


        plt.legend()
        fig.suptitle("Training metrics")
        plt.savefig("plots/metrics.pdf")
        plt.legend()
        plt.clf()
        plt.close(fig)


    def dnn_output_vs_mt(self):
        x_binning = BinningDict["mt"]
        bin_centers = x_binning[:-1]+(x_binning[1:]-x_binning[:-1])/2.0

        is_signal = self._dataset.loc[:, "event_id"] == 0
        signal_mt = self._dataset.loc[is_signal, "TransverseMass"]
        signal_dnn = self._dataset.loc[is_signal, "predicted_values"]
        bkg_mt = self._dataset.loc[~is_signal, "TransverseMass"]
        bkg_dnn = self._dataset.loc[~is_signal, "predicted_values"]

        sig_bin_means = -np.ones(len(bin_centers))
        sig_bin_std = np.zeros(len(bin_centers))
        bkg_bin_means = -np.ones(len(bin_centers))
        bkg_bin_std = np.zeros(len(bin_centers))

        sig_indices = np.digitize(signal_mt, x_binning)
        sig_indices[sig_indices==0] = 1
        sig_indices[sig_indices==len(x_binning)] = len(x_binning)-1
        sig_indices = sig_indices - 1
        bkg_indices = np.digitize(bkg_mt, x_binning)
        bkg_indices[bkg_indices==0] = 1
        bkg_indices[bkg_indices==len(x_binning)] = len(x_binning)-1
        bkg_indices = bkg_indices - 1

        for i in np.unique(sig_indices):
            sig_bin_means[i], sig_bin_std[i] = norm.fit(signal_dnn[sig_indices == i])

        for i in np.unique(bkg_indices):
            bkg_bin_means[i], bkg_bin_std[i] = norm.fit(bkg_dnn[bkg_indices == i])

        sig_up = np.add(sig_bin_means, sig_bin_std/2.0)
        sig_down = np.add(sig_bin_means, -sig_bin_std/2.0)
        bkg_up = np.add(bkg_bin_means, bkg_bin_std/2.0)
        bkg_down = np.add(bkg_bin_means, -bkg_bin_std/2.0)

        fig, (ax_0, ax_1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8,6))
        self._plot_class(sig_bin_means, sig_up, sig_down, bin_centers, x_binning, ax_0, SIG_COLOR, "signal")
        self._plot_class(bkg_bin_means, bkg_up, bkg_down, bin_centers, x_binning, ax_0, BKG_COLOR, "background")

        ax_0.set_ylim(-0.05, 1.40)
        ax_0.set_xlim(x_binning[1], x_binning[-1])
        ax_0.set_xscale('log')
        ax_0.set_ylabel("Mean DNN output")
        ax_0.spines['top'].set_visible(False)
        ax_0.spines['right'].set_visible(False)
        ax_0.spines['left'].set_visible(False)
        ax_0.spines['bottom'].set_visible(False)
        ax_0.grid(False, axis='x')
        locs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax_0.set_yticks(locs)
        ax_0.legend(loc='upper right')

        ax_1.hist((bkg_mt, signal_mt), bins=x_binning, label=["background", "signal"], color=[BKG_COLOR, SIG_COLOR], stacked=True, edgecolor='k', linewidth=1.0, alpha=0.7)
        ax_1.set_xlabel("m$_{T}$")
        ax_1.set_ylabel("Number of events/bin")
        ax_1.set_yscale('log')
        ax_1.spines['top'].set_visible(False)
        ax_1.spines['right'].set_visible(False)
        ax_1.spines['left'].set_visible(False)
        ax_1.spines['bottom'].set_visible(False)
        ax_1.grid(False, axis='x')

        fig.suptitle("DNN output w.r.t. m$_{T}$")

        plt.savefig("plots/dnn_vs_mt.pdf")
        plt.clf()
        plt.close(fig)


    def _plot_class(self, bin_means, bin_ups, bin_downs, bin_centers, x_binning, axes, color, label):
        non_zero_points = (bin_means > -1)
        bin_means = bin_means[non_zero_points]
        bin_ups = bin_ups[non_zero_points]
        bin_downs = bin_downs[non_zero_points]
        bin_centers = bin_centers[non_zero_points]
        bin_centers_refined = np.linspace(x_binning[0], x_binning[-1], num=1000, endpoint=True)

        f_inter_up = interp1d(np.append(np.insert(bin_centers, 0, x_binning[0]), x_binning[-1]), np.append(np.insert(bin_ups, 0, bin_ups[0]), bin_ups[-1]), kind='quadratic')
        f_inter_down = interp1d(np.append(np.insert(bin_centers, 0, x_binning[0]), x_binning[-1]), np.append(np.insert(bin_downs, 0, bin_downs[0]), bin_downs[-1]), kind='quadratic')

        axes.fill_between(bin_centers_refined, f_inter_up(bin_centers_refined), f_inter_down(bin_centers_refined), alpha=0.6, color=color)
        axes.plot(bin_centers_refined, f_inter_up(bin_centers_refined), c=color, linewidth=1.0, alpha=0.7)
        axes.plot(bin_centers_refined, f_inter_down(bin_centers_refined), c=color, linewidth=1.0, alpha=0.7)
        axes.scatter(bin_centers, bin_means, c=color, linewidth=1.0, edgecolors='k', label=label)
