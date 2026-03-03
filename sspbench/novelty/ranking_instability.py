"""
Ranking instability metric based on the AutoBencher paper.

Measures how much a new benchmark disrupts the model rankings established
by existing (static) leaderboards.

    NOVELTY(D_c, D_prev, M) = 1 − RANKCORR(v̂_c, v_c)

where v̂_c is the OLS-predicted accuracy vector from existing leaderboard
scores V_prev, and RANKCORR is Spearman's rank correlation.

High novelty → the new benchmark reveals new information about model
capabilities that cannot be predicted from existing benchmarks.

Reference:
    AutoBencher (Li et al.), Section "Novelty" in the multi-objective
    benchmark optimisation framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple


def load_static_leaderboard(
    csv_path: str,
    model_name_col: str = "Model Name",
    score_col: str = "Hallucination",
) -> pd.DataFrame:
    """Load a static leaderboard CSV.

    Returns a DataFrame with columns ``model`` (normalised name, slash format)
    and ``score`` (float, 0-100 scale).
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={model_name_col: "model", score_col: "score"})
    df = df[["model", "score"]].dropna()
    df["score"] = df["score"].astype(float)
    return df


def _normalise_model_name(name: str) -> str:
    """Normalise model names so both ``meta-llama/Llama-3.2-3B-Instruct``
    and ``meta-llama_Llama-3.2-3B-Instruct`` map to the same key.

    Convention: replace the *first* ``_`` that separates org from model with
    ``/``, keep everything else.  If already has ``/``, leave it.
    Known aliases (evaluation name → leaderboard name) are applied first.
    """
    # Known aliases where eval names differ from leaderboard names
    _ALIASES = {
        "google/gemma-2-2b-it-gptoss": "google/gemma-2-2b-it",
    }

    if "/" not in name:
        # org_model → org/model  (first underscore only)
        parts = name.split("_", 1)
        if len(parts) == 2:
            name = f"{parts[0]}/{parts[1]}"

    return _ALIASES.get(name, name)


def build_accuracy_vector(
    question_model_results: Dict[str, Dict[str, str]],
    question_ids: List[str],
    models: List[str],
    correct_label: str = "Correct",
) -> np.ndarray:
    """Compute the per-model accuracy vector for a subset of questions.

    Parameters
    ----------
    question_model_results : dict
        ``{question_id: {model_name: label}}`` where label ∈ {"Correct", "Incorrect", "RtA"}.
    question_ids : list[str]
        Subset of question IDs to aggregate over.
    models : list[str]
        Ordered list of model names (as they appear in *question_model_results*).
    correct_label : str
        The label that indicates a correct answer.

    Returns
    -------
    np.ndarray of shape ``(len(models),)`` with accuracy ∈ [0, 1].
    """
    qid_set = set(question_ids)
    counts = np.zeros(len(models))
    totals = np.zeros(len(models))

    for qid in qid_set:
        labels = question_model_results.get(qid, {})
        for i, model in enumerate(models):
            if model in labels:
                totals[i] += 1
                if labels[model] == correct_label:
                    counts[i] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(totals > 0, counts / totals, 0.0)
    return acc


def compute_ranking_instability(
    v_new: np.ndarray,
    V_prev: np.ndarray,
) -> Dict[str, float]:
    """Compute the AutoBencher novelty (ranking instability) metric.

    Parameters
    ----------
    v_new : np.ndarray, shape ``(M,)``
        Accuracy vector of the **new** benchmark for M models.
    V_prev : np.ndarray, shape ``(M, N)``
        Accuracy matrix from N **existing** benchmarks for the same M models.
        Each column is a prior benchmark's accuracy vector.

    Returns
    -------
    dict with keys:
        ``ranking_instability`` : float in [0, 2]
            1 − Spearman rank-correlation(v̂, v).  High = more instability.
        ``rank_corr`` : float
            Raw Spearman correlation.
        ``rank_corr_pvalue`` : float
            Two-tailed p-value of the Spearman test.
        ``residual_norm`` : float
            ‖v − v̂‖₂  (how far the OLS prediction is from reality).
    """
    M = v_new.shape[0]
    assert V_prev.shape[0] == M, (
        f"Model count mismatch: v_new has {M}, V_prev has {V_prev.shape[0]} rows"
    )

    # OLS: v̂ = V_prev @ θ* + b
    # Add intercept column (column of ones)
    X = np.column_stack([V_prev, np.ones(M)])

    # Least-squares solve:  min ‖Xw − v_new‖²
    result = np.linalg.lstsq(X, v_new, rcond=None)
    w = result[0]
    v_hat = X @ w

    # Spearman rank correlation between predicted and actual
    corr, pvalue = spearmanr(v_hat, v_new)

    # Handle edge case where all values are identical (correlation undefined)
    if np.isnan(corr):
        corr = 1.0
        pvalue = 1.0

    residual_norm = float(np.linalg.norm(v_new - v_hat))

    return {
        "ranking_instability": 1.0 - corr,
        "rank_corr": float(corr),
        "rank_corr_pvalue": float(pvalue),
        "residual_norm": residual_norm,
    }


def compute_ranking_instability_from_leaderboards(
    v_new: np.ndarray,
    models: List[str],
    leaderboard_dfs: List[pd.DataFrame],
) -> Dict[str, float]:
    """Convenience wrapper that aligns model names between the new accuracy
    vector and one or more static leaderboard DataFrames.

    Parameters
    ----------
    v_new : np.ndarray, shape ``(M,)``
        Per-model accuracy on the new benchmark.
    models : list[str]
        Model names corresponding to *v_new* (evaluation-format names,
        e.g. ``meta-llama_Llama-3.2-3B-Instruct``).
    leaderboard_dfs : list[pd.DataFrame]
        Each DF must have columns ``model`` (org/name) and ``score``.

    Returns
    -------
    Same dict as :func:`compute_ranking_instability`, plus:
        ``matched_models`` : int — number of models matched.
    """
    # Build normalised name → index in v_new
    norm_to_idx = {}
    for i, m in enumerate(models):
        norm_to_idx[_normalise_model_name(m)] = i

    # For each leaderboard, build a column of scores aligned to the common model set
    # First pass: find common models across all leaderboards AND v_new
    common_models: Optional[set] = None
    for lb_df in leaderboard_dfs:
        lb_norms = {_normalise_model_name(m) for m in lb_df["model"]}
        if common_models is None:
            common_models = lb_norms & set(norm_to_idx.keys())
        else:
            common_models &= lb_norms

    if not common_models or len(common_models) < 3:
        return {
            "ranking_instability": float("nan"),
            "rank_corr": float("nan"),
            "rank_corr_pvalue": float("nan"),
            "residual_norm": float("nan"),
            "matched_models": len(common_models) if common_models else 0,
        }

    # Stable ordering
    common_list = sorted(common_models)
    indices = [norm_to_idx[m] for m in common_list]

    v_sub = v_new[indices]

    # Build V_prev columns
    cols = []
    for lb_df in leaderboard_dfs:
        lb_map = {_normalise_model_name(m): s for m, s in zip(lb_df["model"], lb_df["score"])}
        col = np.array([lb_map[m] / 100.0 for m in common_list])  # normalise to [0, 1]
        cols.append(col)

    V_prev = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)

    result = compute_ranking_instability(v_sub, V_prev)
    result["matched_models"] = len(common_list)
    return result


def per_question_ranking_instability_contribution(
    question_model_results: Dict[str, Dict[str, str]],
    question_ids: List[str],
    models: List[str],
    leaderboard_dfs: List[pd.DataFrame],
    correct_label: str = "Correct",
) -> pd.DataFrame:
    """Estimate each question's marginal contribution to ranking instability
    via leave-one-out: for each question *q*, compute the instability of the
    full set minus *q*, and attribute the difference.

    Parameters
    ----------
    question_model_results : dict
        ``{question_id: {model_name: label}}``.
    question_ids : list[str]
        All candidate question IDs.
    models : list[str]
        Model names in evaluation format.
    leaderboard_dfs : list[pd.DataFrame]
        Static leaderboard(s) with ``model`` and ``score`` columns.

    Returns
    -------
    pd.DataFrame with columns ``question_id``, ``instability_contribution``
    (positive = this question *increases* instability when included).
    """
    # Full-set instability
    v_full = build_accuracy_vector(question_model_results, question_ids, models, correct_label)
    full_result = compute_ranking_instability_from_leaderboards(v_full, models, leaderboard_dfs)
    full_instability = full_result["ranking_instability"]

    rows = []
    for qid in question_ids:
        remaining = [q for q in question_ids if q != qid]
        v_loo = build_accuracy_vector(question_model_results, remaining, models, correct_label)
        loo_result = compute_ranking_instability_from_leaderboards(v_loo, models, leaderboard_dfs)
        loo_instability = loo_result["ranking_instability"]

        # If removing this question *decreases* instability, the question contributes positively
        contribution = full_instability - loo_instability
        rows.append({"question_id": qid, "instability_contribution": contribution})

    return pd.DataFrame(rows)
