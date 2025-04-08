from typing import Union, Tuple, List, Dict
from model_equality_testing.pvalue import (
    EmpiricalPvalueCalculator,
    one_sample_parametric_bootstrap_pvalue,
    two_sample_parametric_bootstrap_pvalue,
    two_sample_permutation_pvalue,
)
from model_equality_testing.tests import IMPLEMENTED_TESTS
from model_equality_testing.distribution import (
    CompletionSample,
    DistributionFromDataset,
)


def _noop_pvalue(*args, **kwargs):
    return 1.0


def run_goodness_of_fit_test(
    sample: CompletionSample,
    null_dist: DistributionFromDataset,
    get_pvalue: Union[callable, EmpiricalPvalueCalculator] = None,
    pvalue_type: str = "parametric_bootstrap",
    stat_type: str = "g_squared",
    b=1000,
    **kwargs,
) -> Tuple[float, float]:
    """
    Tests whether the sample is drawn from the null distribution
    Args:
        sample: CompletionSample
        null_dist: DistributionFromDataset
        get_pvalue: callable or EmpiricalPvalueCalculator
            Given a test statistic, returns the p-value
            The function should take in one argument (a float, np.ndarray, or torch.Tensor)
            representing the observed statistic, and it should return a float (the pvalue).
        pvalue_type: str
            If get_pvalue is None, how to compute the p-value
        stat_type: str
            Which test statistic to compute
        b: int
            Number of bootstrap samples if pvalue_type is "parametric_bootstrap"
        kwargs
            Additional arguments to pass to the test statistic function
    Returns:
        pvalue: float
        statistic: float
    """
    if get_pvalue is None:
        if pvalue_type == "parametric_bootstrap":
            get_pvalue = one_sample_parametric_bootstrap_pvalue(
                null_dist=null_dist,
                n=sample.N,
                b=b,
                return_stats=False,
                stat_type=stat_type,
            )
        elif pvalue_type == "dummy":
            get_pvalue = _noop_pvalue
        else:
            raise ValueError("Unrecognized p-value type")

    statistic = IMPLEMENTED_TESTS[stat_type](sample, null_dist, **kwargs)
    pvalue = get_pvalue(statistic)
    if not isinstance(pvalue, float):
        pvalue = pvalue.item()
    return (pvalue, statistic)


def run_two_sample_test(
    sample: CompletionSample,
    other_sample: CompletionSample,
    null_dist: DistributionFromDataset = None,
    get_pvalue: Union[callable, EmpiricalPvalueCalculator] = None,
    pvalue_type: str = "permutation_pvalue",
    stat_type: str = "two_sample_L2",
    b=1000,
    **kwargs,
) -> Tuple[float, float]:
    """
    Tests whether the samples are drawn from the same distribution
    Args:
        sample: CompletionSample
        other_sample: CompletionSample
        null_dist: DistributionFromDataset
        get_pvalue: callable or EmpiricalPvalueCalculator
            Given a test statistic, returns the p-value
            The function should take in one argument (a float, np.ndarray, or torch.Tensor)
            representing the observed statistic, and it should return a float (the pvalue).
        pvalue_type: str
            If get_pvalue is None, how to compute the p-value
        stat_type: str
            Which test statistic to compute
        b: int
            Number of bootstrap samples if pvalue_type is "parametric_bootstrap"
        kwargs
            Additional arguments to pass to the test statistic function
    Returns:
        pvalue: float
        statistic: float
    """
    if get_pvalue is None:
        if pvalue_type == "permutation_pvalue":
            get_pvalue = two_sample_permutation_pvalue(
                sample, other_sample, b=b, stat_type=stat_type, **kwargs
            )
        elif pvalue_type == "parametric_bootstrap":
            assert (
                null_dist is not None
            ), "Must provide null distribution for parametric bootstrap"
            get_pvalue = two_sample_parametric_bootstrap_pvalue(
                null_dist=null_dist,
                n1=sample.N,
                n2=other_sample.N,
                b=b,
                stat_type=stat_type,
                **kwargs,
            )
        elif pvalue_type == "dummy":
            get_pvalue = _noop_pvalue
        else:
            raise ValueError("Unrecognized p-value type")

    statistic = IMPLEMENTED_TESTS[stat_type](sample, other_sample, **kwargs)
    pvalue = get_pvalue(statistic)
    if not isinstance(pvalue, float):
        pvalue = pvalue.item()
    return (pvalue, statistic)
