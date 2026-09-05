"""Statistical Significance Testing for Model and Ensemble Comparisons."""
from typing import Dict, Tuple, Union
import numpy as np
from scipy import stats


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    continuity_correction: bool = True,
) -> Dict[str, Union[float, str, bool]]:
    """Perform McNemar's test for comparing two binary classifiers.

    Tests whether the disagreement proportions between Model A and Model B are statistically significant.

    Args:
        y_true: Ground truth binary labels.
        y_pred_a: Binary predictions from Model A.
        y_pred_b: Binary predictions from Model B.
        continuity_correction: Apply Edwards continuity correction.

    Returns:
        Dict containing statistic, p_value, and significance conclusion.
    """
    y_true = np.array(y_true).astype(int)
    y_pred_a = np.array(y_pred_a).astype(int)
    y_pred_b = np.array(y_pred_b).astype(int)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency matrix
    # n00: both correct
    # n01: A correct, B wrong
    # n10: A wrong, B correct
    # n11: both wrong
    b = np.sum(correct_a & ~correct_b) # A correct, B wrong
    c = np.sum(~correct_a & correct_b) # A wrong, B correct

    if (b + c) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "b_count": int(b),
            "c_count": int(c),
            "conclusion": "No disagreement between models.",
        }

    # Chi-square statistic with optional Edwards continuity correction
    if continuity_correction:
        statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)

    p_value = 1.0 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
        "b_count": int(b),
        "c_count": int(c),
        "conclusion": (
            "Statistically significant difference (p < 0.05)"
            if p_value < 0.05
            else "Difference is not statistically significant (p >= 0.05)"
        ),
    }


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict[str, Union[float, bool, str]]:
    """Perform paired t-test over cross-validation folds or bootstrapped batches."""
    res = stats.ttest_rel(scores_a, scores_b)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_at_05": bool(res.pvalue < 0.05),
        "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
    }
