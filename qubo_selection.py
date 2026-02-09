"""QUBO utilities for selecting rejected applicants to label.

This module builds a QUBO that balances ensemble uncertainty with diversity
while enforcing a fixed labeling budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


DisagreementMethod = Literal["variance", "vote_entropy"]
SimilarityMetric = Literal["cosine", "rbf"]


@dataclass(frozen=True)
class QuboTerms:
    """Linear and quadratic coefficients for a QUBO objective."""

    linear: np.ndarray
    quadratic: np.ndarray


def compute_disagreement(
    probabilities: np.ndarray,
    method: DisagreementMethod = "variance",
) -> np.ndarray:
    """Compute per-sample ensemble disagreement.

    Args:
        probabilities: Array of shape (n_samples, n_models) with per-model
            predicted probabilities for the positive class.
        method: Disagreement score type. "variance" uses model variance of
            probabilities; "vote_entropy" uses entropy of rounded votes.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape (n_samples, n_models)")

    if method == "variance":
        return np.var(probabilities, axis=1)

    if method == "vote_entropy":
        votes = (probabilities >= 0.5).astype(int)
        vote_rate = votes.mean(axis=1)
        vote_rate = np.clip(vote_rate, 1e-8, 1 - 1e-8)
        return -(vote_rate * np.log(vote_rate) + (1 - vote_rate) * np.log(1 - vote_rate))

    raise ValueError(f"Unsupported disagreement method: {method}")


def compute_similarity(
    features: np.ndarray,
    metric: SimilarityMetric = "cosine",
    gamma: float | None = None,
) -> np.ndarray:
    """Compute a similarity matrix for candidate rejects.

    Args:
        features: Array of shape (n_samples, n_features).
        metric: Similarity metric name.
        gamma: RBF kernel parameter. Defaults to 1 / n_features when None.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D array of shape (n_samples, n_features)")

    if metric == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = features / norms
        return np.clip(normalized @ normalized.T, 0.0, 1.0)

    if metric == "rbf":
        if gamma is None:
            gamma = 1.0 / features.shape[1]
        squared_norms = np.sum(features**2, axis=1, keepdims=True)
        sq_dists = squared_norms + squared_norms.T - 2 * (features @ features.T)
        return np.exp(-gamma * np.maximum(sq_dists, 0.0))

    raise ValueError(f"Unsupported similarity metric: {metric}")


def build_qubo(
    utilities: np.ndarray,
    similarities: np.ndarray,
    budget: int,
    diversity_lambda: float,
    penalty: float,
) -> QuboTerms:
    """Build QUBO coefficients for the reject selection problem.

    Args:
        utilities: Per-sample utility scores u_i.
        similarities: Similarity matrix s_ij.
        budget: Number of rejects to label this round.
        diversity_lambda: Weight on similarity penalty.
        penalty: Penalty weight for the budget constraint.
    """
    utilities = np.asarray(utilities, dtype=float)
    similarities = np.asarray(similarities, dtype=float)

    if utilities.ndim != 1:
        raise ValueError("utilities must be a 1D array of shape (n_samples,)")
    if similarities.shape != (utilities.size, utilities.size):
        raise ValueError("similarities must be a square matrix matching utilities length")
    if budget <= 0:
        raise ValueError("budget must be a positive integer")

    linear = -utilities + penalty * (1 - 2 * budget)
    quadratic = diversity_lambda * similarities + 2 * penalty
    np.fill_diagonal(quadratic, 0.0)

    return QuboTerms(linear=linear, quadratic=quadratic)


def qubo_to_upper_triangle(terms: QuboTerms) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upper-triangular indices and values for QUBO solvers."""
    rows, cols = np.triu_indices_from(terms.quadratic, k=1)
    values = terms.quadratic[rows, cols]
    return rows, cols, values
