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

    def plot_psd(self, returns=False):
        fig = plt.figure()

        freqs, powers = self.psd(verbose=False)
        plt.fill_between(freqs, powers.mean(axis=0) - powers.std(axis=0), powers.mean(axis=0) + powers.std(axis=0), alpha=0.2)
        plt.semilogy(freqs, powers.mean(axis=0))

        if returns:
            return fig

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


def build_target_df(annotations, flags=None, threshold=0.5,agg_type='all_ones') -> pd.DataFrame:
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
        targets[flag] = get_target_temp(annotations, flag, threshold, agg_type)

    return targets




def get_target_temp(annotations, flag, threshold=0.5, agg_type='all_ones') -> pd.DataFrame:
    
    """
    Args:
        annotations (pd.DataFrame): dataframe containing targets.
        
        
        flags (dict, list, optional): either list of flag names for which to construct labels or a mapping {flag_name: threshold}. Then it will select each component with its own threshold value.
        If set to None, all flags in annotaions will be selected.
        
        
        agg_type (string): the principle of experts marks aggregation
            all_ones: equal union of all experts marks
            intercept_ones: only overlap of marks is accounted as correct
            weigths_of_ones: all marks are estimated according to their probabilty among experts
            weights: all marks are estimated  according to their probabilty among other marks of each components and among experts
            weigths_with_desicion: marks are estimated  according to their probabilty among other marks of each components and among experts
                                    and the most expected is chosen
            
            
        
    Returns:
        pd.DataFrame: dataframe with flags.
    """
    

    
    def all_ones(ann):
        
        columns=ann.columns
        columns_of_states=columns[3:]
        
        ann_ones=ann.copy()
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones[col] = ann_ones[col].astype(int)
            
            
        ann_ones_group=ann_ones.groupby(['ic_id']).sum()
        ann_ones_ones=ann_ones_group.apply(lambda x: x>0, axis=1)
        
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones_ones[col] = ann_ones_ones[col].astype(int)
            
        return  ann_ones_ones
            
    
    def intercept_ones(ann):
        columns=ann.columns
        columns_of_states=columns[3:]
        
        
        ann_ones=ann.copy()
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones[col] = ann_ones[col].astype(int)
            
        ann_ones_group=ann_ones.groupby(['ic_id']).sum()
        #Здесь пока захардкожено 2
        ann_ones_intercept=ann_ones_group.apply(lambda x: x==2, axis=1)
        
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones_intercept[col] = ann_ones_intercept[col].astype(int)
            
        return  ann_ones_intercept
    
    
    
    def weigths_of_ones(ann):
    
        
        columns=ann.columns
        columns_of_states=columns[3:]
        
        
        ann_ones=ann.copy()
        for i in range(len(columns_of_states)):
            
            col=columns_of_states[i]
            ann_ones[col] = ann_ones[col].astype(int)
            
        ann_ones_group=ann_ones.groupby(['ic_id']).sum()
        #Здесь пока захардкожено 2
        ann_weights_ones=ann_ones_group.apply(lambda x: x/sum(x), axis=1)
        
        return  ann_weights_ones    
    
    
        
    def weights(ann):
        
        columns=ann.columns
        columns_of_states=columns[3:]    
        
        ann_probs=ann.apply(lambda x:  (x[3:]/sum(x[3:]) if sum(x[3:])!=0 else x[3:] ), axis=1)
        ann_probs['ic_id']=ann['ic_id']
        #ann_probs['user_hash']=ann['user_hash']
        
        ann_probs_group=ann_probs.groupby(['ic_id']).sum()
        
        ann_probs_probs=ann_probs_group.apply(lambda x: x/sum(x), axis=1)
        
        return ann_probs_probs
    
    
    
    def weigths_with_desicion(ann):
        
        columns=ann.columns
        columns_of_states=columns[3:]    
        
        ann_probs=ann.apply(lambda x:  (x[3:]/sum(x[3:]) if sum(x[3:])!=0 else x[3:] ), axis=1)
        ann_probs['ic_id']=ann['ic_id']
        #ann_probs['user_hash']=ann['user_hash']
        
        ann_probs_group=ann_probs.groupby(['ic_id']).sum()
        
        ann_probs_probs=ann_probs_group.apply(lambda x: x/sum(x), axis=1)
        ann_probs_probs_with_desicion=ann_probs_probs.apply(lambda x: x==max(x), axis=1)
        
        return ann_probs_probs_with_desicion   
    
    

    if agg_type=='all_ones':
        
        df=all_ones(annotations)
        
    elif agg_type=='intercept_ones':
        
        df=intercept_ones(annotations)
        
    elif agg_type=='weigths_of_ones':
        
        df=weigths_of_ones(annotations)
    
    elif agg_type=='weights':
        
        df=weights(annotations)
        
    elif agg_type=='weigths_with_desicion':
        
        df=weigths_with_desicion(annotations)
    
    
    
    return df[flag]
    
    