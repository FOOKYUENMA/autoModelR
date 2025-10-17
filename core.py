"""
Core functionality for the AutoModelR Python application.

This module provides a suite of functions to preprocess data, select
predictors and outcomes using Lasso, identify mediator and moderator
roles for variables, search for the best ordering of mediators in a
chain, determine the best placement of moderators, and fit simple
path models using ordinary least squares (and optionally SEM if
`semopy` is available).

The functions herein mirror the behaviour of the original R code
provided by the user. They are designed to be robust to messy data
containing mixed types, missing values and variables that should be
ignored.

Author: ChatGPT (OpenAI)
"""

from __future__ import annotations

import itertools
import json
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Attempt to import semopy for structural equation modelling.  If
# unavailable, fallback gracefully; SEM functionality will not be
# exposed.
try:
    import semopy
    from semopy import Model
except ImportError:  # pragma: no cover
    semopy = None


def auto_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Automatically preprocess the input DataFrame.

    The steps mirror those in the R code:

    1. Coerce non-numeric columns to numeric where possible;
       others remain unchanged.
    2. For numeric columns, fill missing values with the column mean.
    3. Standardise numeric columns to mean 0 and unit variance.
    4. Drop numeric columns with zero variance.

    Parameters
    ----------
    data : pandas.DataFrame
        The raw input data with a mixture of numeric and non-numeric
        columns.

    Returns
    -------
    pandas.DataFrame
        The cleaned and standardised data frame.
    """
    df = data.copy()
    # Convert anything that looks numeric to numeric.  Non-numeric
    # entries become NaN. We keep string/factor columns unchanged.
    numeric_cols = []
    for col in df.columns:
        # Check if this column is numeric; attempt coercion otherwise
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # try coercing to numeric; if many values convert, treat as numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            # if at least half of the values are numeric or the type is bool
            non_na_fraction = coerced.notna().mean()
            if non_na_fraction >= 0.5:
                df[col] = coerced
                numeric_cols.append(col)
    # Fill missing and standardise numeric columns
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        mean_val = series.mean(skipna=True)
        series = series.fillna(mean_val)
        # Standardise
        std = series.std(ddof=0)
        if std == 0:
            # constant column; remove later
            df[col] = 0.0
        else:
            df[col] = (series - mean_val) / std
    # Drop numeric columns with zero variance (all zeros after standardising)
    drop_cols = [col for col in numeric_cols if df[col].var(ddof=0) == 0]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def auto_find_main_xy_topn(data: pd.DataFrame, topn: int = 3) -> pd.DataFrame:
    """Automatically find the best outcome and predictor combinations.

    For each numeric column in the data, treat it as the response Y
    and all other numeric columns as potential predictors X. Fit a
    LassoCV model to identify a sparse set of predictors. Compute the
    in-sample R^2 for the model and store the names of the selected
    predictor variables. The results are returned in decreasing order
    of R^2.

    Parameters
    ----------
    data : pandas.DataFrame
        The preprocessed data frame containing only numeric columns.
    topn : int, optional
        The number of top combinations to return, by default 3.

    Returns
    -------
    pandas.DataFrame
        A data frame with columns ['y', 'x', 'r2'] sorted by r2
        descending. 'x' is a comma-separated string of predictor names.
    """
    if data.shape[1] < 2:
        raise ValueError("Need at least two numeric variables to perform selection.")
    num_vars = list(data.columns)
    results = []
    for ycol in num_vars:
        # Response
        y = data[ycol].values
        # Predictors: all other numeric columns
        xcols = [c for c in num_vars if c != ycol]
        if not xcols:
            continue
        X = data[xcols].values
        # Fit Lasso with cross-validation; catch potential warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                model = LassoCV(cv=min(5, len(y)), n_jobs=1).fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                selected = [xcols[i] for i, coef in enumerate(model.coef_) if coef != 0]
                results.append({
                    'y': ycol,
                    'x': ', '.join(selected) if selected else '(none)',
                    'r2': r2
                })
            except Exception as e:
                results.append({
                    'y': ycol,
                    'x': '',
                    'r2': -np.inf
                })
    res_df = pd.DataFrame(results)
    # Sort by r2 descending and return topn
    res_df = res_df.sort_values('r2', ascending=False)
    return res_df.head(topn).reset_index(drop=True)


@dataclass
class RoleResult:
    var: str
    type: str
    p_mediation: float
    p_moderation: float


def _p_value_from_sm(model, param_name: str) -> float:
    """Helper to safely extract p-value for a parameter from a statsmodels summary.

    If the parameter is missing (e.g., due to rank deficiency), returns
    1.0 so that the variable is treated as insignificant.
    """
    try:
        pval = model.pvalues[param_name]
        return float(pval) if not pd.isna(pval) else 1.0
    except Exception:
        return 1.0


def auto_identify_roles(
    data: pd.DataFrame,
    IVs: Iterable[str],
    DV: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Identify mediator and moderator roles among candidate variables.

    For each variable not in IVs or DV, test whether it functions
    primarily as a mediator or as a moderator between the main
    predictor(s) and outcome. A variable is classified as a mediator
    if both the path from IV to the variable and the path from the
    variable to DV are significant, and these p-values are smaller
    than the interaction p-value (moderation effect). Otherwise, if
    the interaction term between IV and the variable is significant,
    classify it as a moderator. Otherwise, label as 'none'.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed numeric DataFrame.
    IVs : Iterable[str]
        Names of predictor (independent) variables.
    DV : str
        Name of response (dependent) variable.
    alpha : float, optional
        Significance threshold for tests, by default 0.05.

    Returns
    -------
    pandas.DataFrame
        Data frame with columns ['var', 'type', 'p_mediation', 'p_moderation'].
    """
    df = data.copy()
    IVs = [iv for iv in IVs if iv in df.columns]
    if DV not in df.columns:
        raise ValueError(f"DV '{DV}' not found in data.")
    candidates = [c for c in df.columns if c not in set(IVs + [DV])]
    results: List[RoleResult] = []
    for v in candidates:
        # Mediator: test IV -> M and M -> DV
        # Use first IV for simplicity; extension to multiple IVs could
        # combine them.
        main_iv = IVs[0] if IVs else None
        if main_iv is None:
            continue
        # Fit M ~ IV
        M = df[v]
        X1 = sm.add_constant(df[main_iv])
        fit_a = sm.OLS(M, X1).fit()
        p_a = _p_value_from_sm(fit_a, main_iv)
        # Fit DV ~ IV + M
        Y = df[DV]
        X2 = sm.add_constant(pd.concat([df[main_iv], df[v]], axis=1))
        fit_b = sm.OLS(Y, X2).fit()
        p_b = _p_value_from_sm(fit_b, v)
        p_med = max(p_a, p_b)
        # Moderator: fit DV ~ IV + Z + IV*Z
        X3 = df[main_iv]
        Z = df[v]
        X_int = X3 * Z
        X_mod = sm.add_constant(pd.concat([X3, Z, X_int.rename('interaction')], axis=1))
        fit_mod = sm.OLS(Y, X_mod).fit()
        p_int = _p_value_from_sm(fit_mod, 'interaction')
        # Classification
        if p_med < alpha and p_med < p_int:
            role_type = 'mediator'
        elif p_int < alpha:
            role_type = 'moderator'
        else:
            role_type = 'none'
        results.append(RoleResult(
            var=v,
            type=role_type,
            p_mediation=p_med,
            p_moderation=p_int
        ))
    res_df = pd.DataFrame([r.__dict__ for r in results])
    return res_df


def auto_mediator_order(
    data: pd.DataFrame,
    IVs: Iterable[str],
    DV: str,
    mediators: Iterable[str],
) -> Tuple[List[str], float]:
    """Search all permutations of mediators to find the best chain.

    The scoring criterion sums the R^2 values of the following models:
    - First mediator ~ IVs
    - Each subsequent mediator ~ IVs + all preceding mediators
    - DV ~ IVs + all mediators

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed numeric data.
    IVs : Iterable[str]
        Independent variables (first IV used for scoring).
    DV : str
        Dependent variable.
    mediators : Iterable[str]
        Candidate mediators to order.

    Returns
    -------
    Tuple[List[str], float]
        The best ordering and its total score.
    """
    IVs = [iv for iv in IVs if iv in data.columns]
    if not mediators:
        return [], float('-inf')
    best_order = list(mediators)
    best_score = float('-inf')
    for perm in itertools.permutations(mediators):
        score = 0.0
        # First mediator ~ IVs
        M1 = data[perm[0]]
        X = sm.add_constant(data[IVs])
        fit = sm.OLS(M1, X).fit()
        score += fit.rsquared
        # Subsequent mediators ~ IVs + previous mediators
        for i in range(1, len(perm)):
            Mi = data[perm[i]]
            prev = list(perm[:i])
            X_chain = sm.add_constant(data[IVs + prev])
            fit = sm.OLS(Mi, X_chain).fit()
            score += fit.rsquared
        # DV ~ IVs + all mediators
        Y = data[DV]
        X_end = sm.add_constant(data[IVs + list(perm)])
        fit = sm.OLS(Y, X_end).fit()
        score += fit.rsquared
        if score > best_score:
            best_score = score
            best_order = list(perm)
    return best_order, best_score


def auto_moderator_position(
    data: pd.DataFrame,
    IVs: Iterable[str],
    DV: str,
    mediators: Iterable[str],
    moderators: Iterable[str],
) -> Tuple[str, float, str]:
    """Determine the best placement of moderators in the path.

    For each moderator, we test three possible interactions:
    1. Moderating the IV → DV path (model: DV ~ IV * mod)
    2. Moderating the IV → mediator path for each mediator
       (model: M ~ IV * mod)
    3. Moderating the mediator → DV path (model: DV ~ M * mod)

    We compute the adjusted R^2 for each model and select the
    combination with the highest value.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed numeric data.
    IVs : Iterable[str]
        Independent variables.
    DV : str
        Dependent variable.
    mediators : Iterable[str]
        Mediators (in order) as returned by auto_mediator_order.
    moderators : Iterable[str]
        Candidate moderators.

    Returns
    -------
    Tuple[str, float, str]
        The best moderator, its score, and a description of the path.
    """
    best_mod = None
    best_score = float('-inf')
    best_path = ''
    main_iv = IVs[0] if IVs else None
    if main_iv is None:
        return None, float('nan'), ''
    Y = data[DV]
    for mod in moderators:
        mod_series = data[mod]
        # 1. moderating IV->DV
        X1 = data[main_iv]
        inter1 = X1 * mod_series
        Xmat1 = sm.add_constant(pd.concat([X1, mod_series, inter1.rename('interaction')], axis=1))
        fit1 = sm.OLS(Y, Xmat1).fit()
        score1 = fit1.rsquared_adj
        if score1 > best_score:
            best_mod, best_score, best_path = mod, score1, f"moderator {mod} on {main_iv} → {DV}"
        # 2. moderating IV->mediator(s)
        for med in mediators:
            M = data[med]
            inter2 = data[main_iv] * mod_series
            Xmat2 = sm.add_constant(pd.concat([data[main_iv], mod_series, inter2.rename('interaction')], axis=1))
            fit2 = sm.OLS(M, Xmat2).fit()
            score2 = fit2.rsquared_adj
            if score2 > best_score:
                best_mod, best_score, best_path = mod, score2, f"moderator {mod} on {main_iv} → {med}"
        # 3. moderating mediator → DV
        for med in mediators:
            M = data[med]
            inter3 = M * mod_series
            Xmat3 = sm.add_constant(pd.concat([M, mod_series, inter3.rename('interaction')], axis=1))
            fit3 = sm.OLS(Y, Xmat3).fit()
            score3 = fit3.rsquared_adj
            if score3 > best_score:
                best_mod, best_score, best_path = mod, score3, f"moderator {mod} on {med} → {DV}"
    return best_mod, best_score, best_path


def auto_path_model_fit(
    data: pd.DataFrame,
    IVs: Iterable[str],
    DV: str,
    mediators: Iterable[str],
    moderator: Optional[str],
    use_sem: bool = False,
) -> Tuple[str, Optional[object]]:
    """Fit a simple path model and return a human-readable summary.

    This function constructs a chain mediation model with an optional
    moderator. If `use_sem` is True and the `semopy` package is
    available, the function will build a structural equation model. If
    semopy is unavailable or use_sem is False, the function falls
    back to fitting separate OLS regressions for each path and
    concatenates their summaries. In both cases, a plain text
    description of the model is returned, along with the fitted model
    object (semopy Model or list of regression results).

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed numeric data.
    IVs : Iterable[str]
        Independent variables (first element used primarily).
    DV : str
        Dependent variable.
    mediators : Iterable[str]
        Ordered list of mediators.
    moderator : Optional[str]
        Name of the moderator variable, if any.
    use_sem : bool, optional
        Whether to attempt an SEM fit via semopy, by default False.

    Returns
    -------
    Tuple[str, Optional[object]]
        A tuple containing a string summarising the model and the
        fitted model object (Model or list of OLS results). The
        summary includes parameter estimates and basic fit indices.
    """
    iv = IVs[0] if IVs else None
    if iv is None:
        return "No IV provided.", None
    # Build model string for SEM
    if use_sem and semopy is not None and mediators:
        model_desc_lines = []
        # chain: M1 ~ iv; M2 ~ M1 + iv; ... ; DV ~ lastM + iv
        for idx, m in enumerate(mediators):
            if idx == 0:
                model_desc_lines.append(f"{m} ~ {iv}")
            else:
                prev = mediators[idx - 1]
                model_desc_lines.append(f"{m} ~ {prev} + {iv}")
        model_desc_lines.append(f"{DV} ~ {mediators[-1]} + {iv}")
        # Add moderator as main effect (not as interaction) for SEM
        if moderator:
            model_desc_lines[-1] = model_desc_lines[-1] + f" + {moderator}"
        model_desc = '\n'.join(model_desc_lines)
        try:
            model = Model(model_desc)
            model.fit(data)
            est = model.inspect(std_est=True)
            fit = semopy.calc_stats(model)
            summary_lines = ["SEM fit via semopy:\n", model_desc, "\n\n", "Parameter estimates:\n", est.to_string(index=False), "\n\n"]
            # Fit indices
            summary_lines.append("Fit indices:\n")
            for k in ["CFI", "RMSEA", "SRMR", "AIC", "BIC"]:
                if k in fit:
                    summary_lines.append(f"{k}: {fit[k]:.3f}\n")
            summary = ''.join(summary_lines)
            return summary, model
        except Exception as e:
            use_sem = False  # fallback
    # Fallback: sequential OLS regression summaries
    results = []
    summary_lines = []
    # If no mediators, simple regression
    if not mediators:
        Y = data[DV]
        X = sm.add_constant(data[iv])
        res = sm.OLS(Y, X).fit()
        summary_lines.append("Simple regression:\n")
        summary_lines.append(str(res.summary()))
        return ''.join(summary_lines), [res]
    # chain regression for mediators
    for idx, m in enumerate(mediators):
        if idx == 0:
            predictors = [iv]
        else:
            predictors = [iv] + mediators[:idx]
        Y = data[m]
        X = sm.add_constant(data[predictors])
        res = sm.OLS(Y, X).fit()
        results.append((f"{m} ~ {', '.join(predictors)}", res))
    # final outcome regression
    predictors = [iv] + mediators
    if moderator:
        predictors.append(moderator)
    Y = data[DV]
    X = sm.add_constant(data[predictors])
    res = sm.OLS(Y, X).fit()
    results.append((f"{DV} ~ {', '.join(predictors)}", res))
    # Format summary
    for desc, res in results:
        summary_lines.append(desc + "\n")
        summary_lines.append(str(res.summary()) + "\n\n")
    return ''.join(summary_lines), results