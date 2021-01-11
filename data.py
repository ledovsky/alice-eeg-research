from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_epochs_from_df

class IC:
    """
    A wrapper that represents the independent component. Contains the signal, weights of channels and the sampling frequency.
    """

    def __init__(self, signal, weigths, freq):
        self.signal = signal
        self.weights = weigths
        self.freq = freq

    def select_weights(self, channels):
        return self.weights[self.weights.index.isin(channels)]

    @lru_cache(maxsize=10)
    def psd(self, **kwargs):
        epochs = get_epochs_from_df(self.signal, self.freq)
        powers, freqs = mne.time_frequency.psd_multitaper(epochs, picks=[0], **kwargs)
        return freqs, powers.mean(axis=1)

    def plot_psd():
        raise NotImplementedError

    def plot_topomap(self, returns=False):
        fig, ax = plt.subplots()

        outlines = 'head'

        res = 64
        contours = 6
        sensors = True
        image_interp = 'bilinear'
        show = True
        extrapolate = 'box'

        border = 0

        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        ten_twenty_montage_channels = {ch.lower(): ch for ch in ten_twenty_montage.ch_names}

        # get channels in format of ten_twenty_montage in right order

        channels_to_use_ = [ten_twenty_montage_channels[ch] for ch in self.weights.index]

        # create Info object to store info
        info = mne.io.meas_info.create_info(channels_to_use_, sfreq=256, ch_types="eeg")

        # using temporary RawArray to apply mongage to info
        mne.io.RawArray(np.zeros((len(channels_to_use_), 1)), info, copy=None, verbose=False).set_montage(ten_twenty_montage)

        # pick channels
        channels_to_use_ = [ch for ch in info.ch_names if ch.lower() in self.weights.index]
        info.pick_channels(channels_to_use_)

        _, pos, _, names, _, sphere, clip_origin = mne.viz.topomap._prepare_topomap_plot(info, 'eeg')

        outlines = mne.viz.topomap._make_head_outlines(sphere, pos, outlines, clip_origin)

        mne.viz.topomap.plot_topomap(
            self.weights, pos, res=res,
            outlines=outlines, contours=contours, sensors=sensors,
            image_interp=image_interp, show=show, extrapolate=extrapolate,
            sphere=sphere, border=border, axes=ax, names=names
        )

        if returns:
            return fig

def read_ic(dir, ic_id, freqs=None):
    path = Path(dir)
    if freqs is None:
        freqs = pd.read_csv(path/'ics.csv')
    data = pd.read_csv(path/f'{ic_id}_data.csv')
    signal = data.groupby('epoch')['value'].apply(np.array).rename('signal')
    weights = pd.read_csv(path/f'{ic_id}_weights.csv', index_col='ch_name')['value'].rename('weights')
    return IC(signal, weights, freqs.loc[freqs['ic_id'] == ic_id, 'sfreq'])



def load_dataset(dir='data'):
    path = Path(dir)
    freqs = pd.read_csv(path/'ics.csv')
    ic_ids = list(freqs['ic_id'])
    data = {ic_id: read_ic(dir, ic_id, freqs) for ic_id in ic_ids}
    annotations = pd.read_csv(path/'annotations_raw.csv')
    return data, annotations


def get_target(annotations, flag, threshold=0.5) -> pd.Series:
    """
    Args:
        annotations (pd.DataFrame): dataframe containing targets. IC column must be named 'ic_id'.
        flag (str): flag name
        threshold (float, optional): if the average label of a component is above this value, the flag is set True. Defaults to 0.5.

    Returns:
        pd.Series: flags for each component.
    """
    return annotations.groupby('ic_id')[flag].mean() > threshold


def get_flag_names(annotations) -> list:
    return annotations.columns[annotations.columns.str.startswith('flag_')]


def build_target_df(annotations, flags=None, threshold=0.5) -> pd.DataFrame:
    """
    Args:
        annotations (pd.DataFrame): dataframe containing targets.
        flags (dict, list, optional): either list of flag names for which to construct labels or a mapping {flag_name: threshold}. Then it will select each component with its own threshold value.
        If set to None, all flags in annotaions will be selected.
        threshold (float, optional): if the average label of a component is above this value, the flag is set True. Defaults to 0.5.

    Returns:
        pd.DataFrame: dataframe with flags.
    """
    targets = pd.DataFrame()

    if flags is None:
        flags = get_flag_names(annotations)
    if not isinstance(flags, dict):
        flags = dict.fromkeys(flags, threshold)

    for flag, threshold in flags.items():
        targets[flag] = get_target(annotations, flag, threshold)

    return targets