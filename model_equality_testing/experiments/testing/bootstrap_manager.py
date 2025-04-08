from typing import Dict, List, Union, Tuple
import numpy as np
import torch
import pickle
from model_equality_testing.pvalue import EmpiricalPvalueCalculator


def load_parametric_bootstrap(path, min_b=None, max_b=None) -> np.ndarray:
    """
    Loads pre-saved test stats from bootstrapping and enforces the correct shapes.
    Output shape is (b, 1, n_stats) where b is the number of bootstraps and n_stats is the number of statistics.
    """
    with open(path, "rb") as f:
        stats = pickle.load(f)
    if stats.ndim == 1:
        stats = np.expand_dims(stats, 0)
    if stats.ndim == 2:
        stats = np.expand_dims(stats, 2)
    if max_b is not None:
        stats = stats[:max_b]
    if min_b is not None:
        assert min_b <= len(
            stats
        ), f"min_b={min_b} is greater than the number of bootstraps {len(stats)}"
    return stats


class BootstrapManager:
    """
    Helper class to manage loading of bootstrapped statistics with different sample sizes.
    """

    def __init__(self, bootstrap_path_template: str, min_b=None, max_b=None):
        """
        Args:
            bootstrap_path_template: str
                A string template that can be filled in with the sample size n
                Example: "cache/parametric_bootstrap_stats/meta-llama-Meta-Llama-3-8B-wikipedia-{n}.pkl"
            min_b: int
                Minimum number of bootstraps to load
            max_b: int
                Maximum number of bootstraps to load
        """
        self._bootstrap_path_template = bootstrap_path_template
        self.min_b = min_b
        self.max_b = max_b
        self._stats = None

    def load(
        self,
        return_stats=False,
        **kwargs,
    ) -> Union[EmpiricalPvalueCalculator, Tuple[np.ndarray, EmpiricalPvalueCalculator]]:
        """
        Loads the bootstrapped statistics for a given sample size n.
        Args:
            n: int or List[int]
                The sample size to load statistics for
            return_stats: bool
                If True, returns the raw statistics as well as the p-value calculator
        Returns:
            EmpiricalPvalueCalculator or Tuple[np.ndarray, EmpiricalPvalueCalculator]
        """
        for k, v in kwargs.items():
            try:
                v = v.item()
            except:
                pass
        self._stats = load_parametric_bootstrap(
            self._bootstrap_path_template.format(**kwargs),
            min_b=self.min_b,
            max_b=self.max_b,
        )

        if return_stats:
            return self._stats, EmpiricalPvalueCalculator(self._stats)
        return EmpiricalPvalueCalculator(self._stats)
