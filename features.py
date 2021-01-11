import numpy as np

from utils import *


def compute_K(ic, mean_shift=False, thres=0.99) -> float:
    """
    Args:
        ic (IC): indepentent component.
        mean_shift (bool, optional): if set to True, the average over epoch is subtracted. Defaults to False.
        thres (float, optional): determines whether to remove the top 1 - thres of values. Defaults to 0.99.

    Returns:
        float: Temporal Kurtosis feature.
    """
    def _epoch_kurtosis(epoch):
        if not mean_shift:
            return np.mean(epoch ** 4) / (np.mean(epoch ** 2) ** 2) - 3
        return np.mean((epoch - epoch.mean()) ** 4) / (epoch.var() ** 2) - 3

    epochs = ic.signal.apply(_epoch_kurtosis)

    if not thres:
        return epochs.mean()
    return trim(epochs, thres).mean()


def compute_MEV(ic, thres=0.99) -> float:
    """
    Args:
        ic (IC): indepentent component.
        thres (float, optional): determines whether to remove the top 1 - thres of values. Defaults to 0.99.

    Returns:
        float: Maximum Epoch Variance feature.
    """
    vars = ic.signal.apply(np.var)

    if not thres:
        return vars.max() / vars.mean()
    return vars.quantile(thres) / trim(vars, thres).mean()


FA = {'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
      'af3', 'af4', 'af7', 'af8',
      'fp1', 'fpz', 'fp2'}
PA = {'cpz', 'cp1', 'cp2', 'cp3', 'cp4', 'cp5', 'cp6',
      'pz', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 't5', 't6', 'p9', 'p10',
      'poz', 'po3', 'po4', 'po7', 'po8',
      'oz', 'o1', 'o2'}
LE = {'f3', 'f5', 'f7', 'f9',
      'af7'}
RE = {'f4', 'f6', 'f8', 'f10',
      'af6'}


def compute_SAD(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Average Difference feature.
    """
    return np.abs(ic.select_weights(FA).mean()) - np.abs(ic.select_weights(PA).mean())


def compute_SVD(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Variance Difference feature.
    """
    return ic.select_weights(FA).var() - ic.select_weights(PA).var()


def compute_SED(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Eye Difference feature.
    """
    return np.abs(ic.select_weights(LE).mean() - ic.select_weights(RE).mean())


def compute_MIF(ic):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Myogenic identification feature.
    """
    freqs, psd = ic.psd(verbose=False)
    mean_psd = psd.mean(axis=0)
    return mean_psd[freqs > 20].sum() / mean_psd.sum()


def compute_CIF(ic):
    # TODO Implement feature. Address low frequency resolution
    raise NotImplementedError


default_features = {'K': compute_K,
                    'MEV': compute_MEV,
                    'SAD': compute_SAD,
                    'SVD': compute_SVD,
                    'SED': compute_SED,
                    'MIF': compute_MIF}


def build_feature_df(data, default=True, custom_features={}):
    feature_df = pd.DataFrame(index=data.keys())
    if default:
        for feature_name, compute_feature in default_features.items():
            feature_df[feature_name] = [compute_feature(ic) for ic in data.values()]

    for feature_name, compute_feature in custom_features.items():
        feature_df[feature_name] = [compute_feature(ic) for ic in data.values()]
    return feature_df