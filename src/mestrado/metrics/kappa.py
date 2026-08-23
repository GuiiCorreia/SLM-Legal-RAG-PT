"""
Fleiss' Kappa for inter-rater agreement.

Used to validate LLM-as-judge consistency across multiple judges or multiple runs.
Used to validate the LLM judge against a second judge (Table 6).

References:
  - Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
    Psychological Bulletin, 76(5), 378-382.
  - Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
    for categorical data. Biometrics, 33(1), 159-174.
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for multiple raters.

    Measures inter-rater agreement for categorical ratings, accounting for
    agreement expected by chance. Used to validate LLM-as-judge consistency.

    Args:
        ratings: Matrix of shape (n_items, n_raters) with ratings.
                 Ratings should be integers 0, 1, 2, ... (categorical).
                 Example: 3 raters rating 5 items on a 3-point scale (0,1,2):
                 [[0, 1, 0],  # item 0
                  [1, 2, 1],  # item 1
                  [2, 1, 2],  # item 2
                  [0, 0, 0],  # item 3
                  [1, 1, 2]]  # item 4

    Returns:
        Kappa score between -1 and 1:
        - 1.0: Perfect agreement
        - 0.0: Agreement equal to chance
        - < 0: Less agreement than expected by chance

    Interpretation (Landis & Koch 1977):
        - 0.81-1.00: Almost perfect agreement
        - 0.61-0.80: Substantial agreement
        - 0.41-0.60: Moderate agreement
        - 0.21-0.40: Fair agreement
        - 0.00-0.20: Slight agreement
        - < 0.00: Poor agreement

    References:
        - Fleiss (1971)
        - Landis & Koch (1977)
    """
    n_items, n_raters = ratings.shape

    if n_items == 0:
        logger.warning("Empty ratings matrix. Returning 0.0")
        return 0.0

    if n_raters == 1:
        logger.warning("Only one rater. Cannot compute kappa. Returning 1.0")
        return 1.0

    # Count category frequencies
    n_categories = int(np.max(ratings)) + 1

    # n_jk = number of raters who assigned category k to item j
    n_jk = np.zeros((n_items, n_categories), dtype=float)
    for j in range(n_items):
        for k in range(n_categories):
            n_jk[j, k] = np.sum(ratings[j, :] == k)

    # p_j = proportion of raters who agreed on item j
    # P_j = (sum_k n_jk^2 - n_raters) / (n_raters * (n_raters - 1))
    P_j = (np.sum(n_jk ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # P_bar = mean proportion of agreement across all items
    P_bar = np.mean(P_j)

    # p_k = proportion of all assignments to category k
    p_k = np.sum(n_jk, axis=0) / (n_items * n_raters)

    # P_e = expected agreement by chance
    P_e = np.sum(p_k ** 2)

    # Avoid division by zero
    if P_e == 1.0:
        logger.warning("All categories equally likely. Kappa is undefined. Returning 0.0")
        return 0.0

    # Fleiss' Kappa
    kappa = (P_bar - P_e) / (1.0 - P_e)

    return float(kappa)


def fleiss_kappa_interpretation(kappa: float) -> str:
    """
    Interpret Fleiss' Kappa score according to Landis & Koch (1977).

    Args:
        kappa: Kappa score between -1 and 1

    Returns:
        Interpretation string
    """
    if kappa >= 0.81:
        return "Almost perfect agreement"
    elif kappa >= 0.61:
        return "Substantial agreement"
    elif kappa >= 0.41:
        return "Moderate agreement"
    elif kappa >= 0.21:
        return "Fair agreement"
    elif kappa >= 0.00:
        return "Slight agreement"
    else:
        return "Poor agreement (less than chance)"


def compute_kappa_for_judges(
    judge_ratings: list[list[float]],
    bins: list[float] = [0.0, 0.33, 0.67, 1.0],
) -> float:
    """
    Compute Fleiss' Kappa for continuous ratings (e.g., 0.0-1.0 LLM scores).

    Converts continuous ratings to categorical bins before computing kappa.

    Args:
        judge_ratings: List of lists, where each inner list is one judge's ratings
                       for all items.
                       Example: [[0.8, 0.7, 0.9], [0.7, 0.8, 0.7], [0.9, 0.7, 0.8]]
                                 (3 judges, 3 items each)
        bins: Bins for discretization. Default: [0.0, 0.33, 0.67, 1.0]
              Creates categories: [0,0.33), [0.33,0.67), [0.67,1.0]

    Returns:
        Kappa score

    Example:
        >>> judges = [[0.8, 0.7, 0.9], [0.7, 0.8, 0.7], [0.9, 0.7, 0.8]]
        >>> kappa = compute_kappa_for_judges(judges)
        >>> print(f"Kappa: {kappa:.3f}")
    """
    if not judge_ratings:
        logger.warning("Empty judge ratings. Returning 0.0")
        return 0.0

    n_judges = len(judge_ratings)
    n_items = len(judge_ratings[0])

    # Validate all judges have same number of ratings
    for i, ratings in enumerate(judge_ratings):
        if len(ratings) != n_items:
            raise ValueError(f"Judge {i} has {len(ratings)} ratings, expected {n_items}")

    # Convert to numpy array: shape (n_items, n_judges)
    ratings_array = np.array(judge_ratings).T

    # Discretize continuous ratings
    categorical_ratings = np.zeros_like(ratings_array, dtype=int)
    for i in range(len(bins) - 1):
        categorical_ratings[(ratings_array >= bins[i]) & (ratings_array < bins[i + 1])] = i

    # Handle edge case: ratings exactly at max bin
    categorical_ratings[ratings_array == bins[-1]] = len(bins) - 2

    return fleiss_kappa(categorical_ratings)


def kappa_validation_summary(kappa: float, n_items: int, n_raters: int) -> dict:
    """
    Generate a summary of kappa validation results.

    Args:
        kappa: Kappa score
        n_items: Number of items rated
        n_raters: Number of raters/judges

    Returns:
        Dictionary with summary information
    """
    interpretation = fleiss_kappa_interpretation(kappa)

    # Target: substantial agreement (kappa >= 0.6) for reliable LLM-as-judge
    meets_target = kappa >= 0.6

    return {
        "kappa": round(kappa, 4),
        "interpretation": interpretation,
        "meets_target": meets_target,
        "target": "substantial agreement (kappa >= 0.6)",
        "n_items": n_items,
        "n_raters": n_raters,
        "status": "PASS" if meets_target else "FAIL",
    }
