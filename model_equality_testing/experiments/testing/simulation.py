import numpy as np
import torch
from model_equality_testing.distribution import DistributionFromDataset
from model_equality_testing.algorithm import (
    run_goodness_of_fit_test,
    run_two_sample_test,
)
from typing import Union, List, Tuple
import tqdm
import gc


def get_power_one_sample(
    null_dist: DistributionFromDataset,
    data_dist: DistributionFromDataset,
    n: int,
    n_simulations=100,
    alpha: float = 0.05,
    pvalue_type: str = "parametric_bootstrap",
    stat_type: str = "g_squared",
    get_pvalue_fn=None,
    return_pvalue: bool = False,
    return_alpha: bool = False,
    return_stat: bool = False,
    **kwargs,
):
    """
    Power analysis for joint (prompt, completion) distribution
    """
    rejections = []
    pvalues = []
    stat = []
    for _ in tqdm.tqdm(range(n_simulations), desc="Power simulation"):
        sample = data_dist.sample(n=n)
        pv, s = run_goodness_of_fit_test(
            sample=sample,
            null_dist=null_dist,
            get_pvalue=get_pvalue_fn,
            pvalue_type=pvalue_type,
            stat_type=stat_type,
            **kwargs,
        )
        stat.append(s)
        pvalues.append(pv)
        rejections.append(int(pv <= alpha))
    power = sum(rejections) / n_simulations
    return_tuple = (power, np.array(rejections, dtype=bool))
    if return_alpha:
        return_tuple += (alpha * np.ones((n_simulations, null_dist.m)),)
    if return_pvalue:
        return_tuple += (torch.from_numpy(np.stack(pvalues)),)
    if return_stat:
        return_tuple += (np.stack(stat),)
    return return_tuple


def get_power_two_sample(
    null_dist: DistributionFromDataset,
    data_dist: DistributionFromDataset,
    n_null: int,
    n_data: int,
    n_simulations=100,
    alpha: float = 0.05,
    pvalue_type: str = "permutation_pvalue",
    stat_type: str = "two_sample_L2",
    get_pvalue_fn=None,
    return_pvalue: bool = False,
    return_alpha: bool = False,
    return_stat: bool = False,
    **kwargs,
):
    """
    Power analysis for joint (prompt, completion) distribution
    """
    rejections = []
    pvalues = []
    stat = []
    for _ in tqdm.tqdm(range(n_simulations), desc="Power simulation"):
        sample_1 = null_dist.sample(n=n_null)
        sample_2 = data_dist.sample(n=n_data)
        pv, s = run_two_sample_test(
            sample=sample_1,
            other_sample=sample_2,
            null_dist=null_dist,
            get_pvalue=get_pvalue_fn,
            pvalue_type=pvalue_type,
            stat_type=stat_type,
            **kwargs,
        )
        pvalues.append(pv)
        stat.append(s)
        rejections.append(int(pv <= alpha))
        del sample_1, sample_2
        gc.collect()
    power = sum(rejections) / n_simulations
    return_tuple = (power, np.array(rejections, dtype=bool))
    if return_alpha:
        return_tuple += (alpha * np.ones((n_simulations, null_dist.m)),)
    if return_pvalue:
        return_tuple += (torch.from_numpy(np.stack(pvalues)),)
    if return_stat:
        return_tuple += (np.stack(stat),)
    return return_tuple


def get_power_two_sample_composite_null(
    null_dist_1: DistributionFromDataset,
    null_dist_2: DistributionFromDataset,
    data_dist: DistributionFromDataset,
    n_null: int,
    n_data: int,
    n_simulations=100,
    alpha: float = 0.05,
    pvalue_type: str = "permutation_pvalue",
    stat_type: str = "two_sample_L2",
    get_pvalue_fn_1=None,
    get_pvalue_fn_2=None,
    return_pvalue: bool = False,
    return_alpha: bool = False,
    return_stat: bool = False,
    **kwargs,
):
    """
    Power analysis for joint (prompt, completion) distribution
    """
    rejections = []
    pvalues = []
    stat = []
    for _ in tqdm.tqdm(range(n_simulations), desc="Power simulation"):
        sample_1 = null_dist_1.sample(n=n_null)
        sample_2 = null_dist_2.sample(n=n_null)
        sample_3 = data_dist.sample(n=n_data)
        pv1, s1 = run_two_sample_test(
            sample=sample_1,
            other_sample=sample_3,
            get_pvalue=get_pvalue_fn_1,
            pvalue_type=pvalue_type,
            stat_type=stat_type,
            **kwargs,
        )
        pv2, s2 = run_two_sample_test(
            sample=sample_2,
            other_sample=sample_3,
            get_pvalue=get_pvalue_fn_2,
            pvalue_type=pvalue_type,
            stat_type=stat_type,
            **kwargs,
        )
        rejections.append(int((pv1 <= alpha) and (pv2 <= alpha)))
        pvalues.append((pv1, pv2))
        stat.append((s1, s2))
        del sample_1, sample_2
        gc.collect()
    power = sum(rejections) / n_simulations
    return_tuple = (power, np.array(rejections, dtype=bool))
    if return_alpha:
        return_tuple += (alpha * np.ones((n_simulations, null_dist_1.m)),)
    if return_pvalue:
        return_tuple += (torch.from_numpy(np.stack(pvalues)),)
    if return_stat:
        return_tuple += (np.stack(stat),)
    return return_tuple
