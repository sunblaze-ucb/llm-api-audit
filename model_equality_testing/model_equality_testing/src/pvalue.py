import numpy as np
import torch
from typing import Union, Tuple, List, Dict
from model_equality_testing.tests import IMPLEMENTED_TESTS
from model_equality_testing.distribution import (
    CompletionSample,
    DistributionFromDataset,
)
import tqdm
import matplotlib.pyplot as plt


def _plot_empirical_distribution(stats, ax=None, label="", **kwargs):
    """
    Given a numpy array of test stats, plots the empirical distribution using a histogram
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.hist(stats, bins="auto", **kwargs)  # Adjust bins as needed
    ax.set_xlabel("Test Statistic")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f'Distribution of Test Statistic {"(" + label + ")" if len(label) else ""}'
    )
    return ax


##############################################
# Functions to simulate and compute p-values
##############################################


class EmpiricalPvalueCalculator:
    """
    Given an empirical sample of test statitics, provides a callable that returns the p-value of an observed test statistic.
    """

    def __init__(self, observed_stats: np.ndarray):
        """
        Args:
            observed_stats: a numpy array of shape (b,)
                where b is the number of bootstrap samples
        """
        self.stats = observed_stats

    def __call__(self, obs_stat: Union[float, np.ndarray, torch.Tensor]) -> float:
        # handle obs_stat: make sure it's a float
        if isinstance(obs_stat, (torch.Tensor, np.ndarray)):
            obs_stat = obs_stat.item()

        # compare to self.stats and average across the batch dimension (b)
        return np.mean((self.stats >= obs_stat), axis=0).item()


def one_sample_parametric_bootstrap_pvalue(
    null_dist: DistributionFromDataset,
    n: int,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="g_squared",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly drawing samples
    from the null distribution and computing the test statistic.
    Args:
        null_dist: a distribution object from which to draw samples
        n: the size of the sample to take
        b: the number of times to draw samples and compute the test statistic
        plot: whether to plot the empirical distribution of the test statistics
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
        stat_type: the type of test statistic to compute as a string.
            Must be a key in IMPLEMENTED_TESTS
        **kwargs: additional arguments to pass to the test computation function
    """
    stats = []
    for _ in tqdm.tqdm(range(b), desc="Parametric bootstrap"):
        bootstrap_sample = null_dist.sample(n=n)
        stat = IMPLEMENTED_TESTS[stat_type](bootstrap_sample, null_dist, **kwargs)
        stats.append(stat)
    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    # plot the empirical distribution
    if plot:
        b, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    return get_pvalue


def two_sample_parametric_bootstrap_pvalue(
    null_dist: DistributionFromDataset,
    n1: int,
    n2: int,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="two_sample_L2",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly drawing samples
    from the null distribution and computing the test statistic.
    Args:
        null_dist: a distribution object from which to draw samples
        n1: the size of the first sample to take
        n2: the size of the second sample to take
        b: the number of times to draw samples and compute the test statistic
        plot: whether to plot the empirical distribution of the test statistics
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
        stat_type: the type of test statistic to compute as a string.
            Must be a key in IMPLEMENTED_TESTS
        **kwargs: additional arguments to pass to the test computation function
    """
    stats = []
    for _ in tqdm.tqdm(range(b), desc="Parametric bootstrap"):
        sample1 = null_dist.sample(n=n1)
        sample2 = null_dist.sample(n=n2)
        stat = IMPLEMENTED_TESTS[stat_type](sample1, sample2, **kwargs)
        stats.append(stat)
    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    # plot the empirical distribution
    if plot:
        b, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    return get_pvalue


def two_sample_permutation_pvalue(
    sample1: CompletionSample,
    sample2: CompletionSample,
    b=1000,
    plot=False,
    return_stats=False,
    stat_type="two_sample_L2",
    **kwargs,
) -> Union[EmpiricalPvalueCalculator, Tuple[EmpiricalPvalueCalculator, np.ndarray]]:
    """
    Simulates the empirical distribution of the test statistic by repeatedly permuting the labels
    of the samples and computing the test statistic.
    Args:
        sample1: the first sample
        sample2: the second sample
        b: the number of times to draw samples and compute the test statistic
        plot: whether to plot the empirical distribution of the test statistics
        return_stats: whether to return the raw test statistics, in addition to
            the p-value calculator
        stat_type: the type of test statistic to compute as a string.
            Must be a key in IMPLEMENTED_TESTS
        **kwargs: additional arguments to pass to the test computation function
    """
    stats = []
    all_samples = torch.cat(
        [
            sample1.sequences,
            sample2.sequences,
        ],
        dim=0,
    )
    for _ in tqdm.tqdm(range(b), desc="Permutation bootstrap"):
        ix = torch.randperm(len(all_samples))
        permuted_sample1 = CompletionSample(
            prompts=all_samples[ix][: sample1.N, 0],
            completions=all_samples[ix][: sample1.N, 1:],
            m=sample1.m,
        )
        permuted_sample2 = CompletionSample(
            prompts=all_samples[ix][sample1.N :, 0],
            completions=all_samples[ix][sample1.N :, 1:],
            m=sample1.m,
        )

        stat = IMPLEMENTED_TESTS[stat_type](
            permuted_sample1, permuted_sample2, **kwargs
        )
        stats.append(stat)

    stats = np.array(stats)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 1)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)

    # plot the empirical distribution
    if plot:
        b, m, nstats = stats.shape
        assert m == 1, "Incorrect shape for plotting"
        for i in range(nstats):
            _plot_empirical_distribution(stats[:, :, i], label=f"{stat_type} dim {i}")

    get_pvalue = EmpiricalPvalueCalculator(stats)
    if return_stats:
        return get_pvalue, stats
    del stats
    return get_pvalue


###### map from name to function ######

IMPLEMENTED_PVALUES = {
    "one_sample_parametric_bootstrap": one_sample_parametric_bootstrap_pvalue,
    "two_sample_parametric_bootstrap": two_sample_parametric_bootstrap_pvalue,
    "two_sample_permutation": two_sample_permutation_pvalue,
}
