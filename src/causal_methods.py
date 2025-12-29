"""
Reusable causal inference methods
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class PropensityScoreMatching:
    """
    Propensity Score Matching for causal effect estimation
    """

    def __init__(self, caliper=0.1):
        """
        Args:
            caliper: Maximum distance for matching (in std dev of propensity scores)
        """
        self.caliper = caliper
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.matched_pairs = None

    def fit(self, X, treatment):
        """
        Estimate propensity scores

        Args:
            X: Covariates (features)
            treatment: Binary treatment indicator (0/1)
        """
        self.propensity_model.fit(X, treatment)
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        return propensity_scores

    def match(self, X, treatment, propensity_scores=None):
        """
        Match treated units to control units based on propensity scores

        Returns:
            DataFrame with matched pairs
        """
        if propensity_scores is None:
            propensity_scores = self.fit(X, treatment)

        # Separate treated and control
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        treated_scores = propensity_scores[treated_idx].reshape(-1, 1)
        control_scores = propensity_scores[control_idx].reshape(-1, 1)

        # Find nearest neighbor for each treated unit
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(control_scores)

        distances, indices = nn.kneighbors(treated_scores)

        # Apply caliper (maximum allowed distance)
        caliper_dist = self.caliper * np.std(propensity_scores)
        valid_matches = distances.flatten() <= caliper_dist

        # Create matched pairs
        matched_pairs = []
        for i, (treated_id, control_id, dist, valid) in enumerate(
            zip(
                treated_idx,
                control_idx[indices.flatten()],
                distances.flatten(),
                valid_matches,
            )
        ):
            if valid:
                matched_pairs.append(
                    {
                        "treated_idx": treated_id,
                        "control_idx": control_id,
                        "distance": dist,
                        "treated_ps": propensity_scores[treated_id],
                        "control_ps": propensity_scores[control_id],
                    }
                )

        self.matched_pairs = pd.DataFrame(matched_pairs)

        print(
            f"Matched {len(self.matched_pairs)} pairs ({len(self.matched_pairs) / len(treated_idx) * 100:.1f}% of treated units)"
        )
        print(
            f"Dropped {len(treated_idx) - len(self.matched_pairs)} treated units (outside caliper)"
        )

        return self.matched_pairs

    def estimate_ate(self, outcomes, matched_pairs=None):
        """
        Estimate Average Treatment Effect on matched sample

        Args:
            outcomes: Outcome variable (e.g., conversion, revenue)
            matched_pairs: DataFrame of matched pairs (uses self.matched_pairs if None)
        """
        if matched_pairs is None:
            matched_pairs = self.matched_pairs

        if matched_pairs is None:
            raise ValueError("Must run match() first or provide matched_pairs")

        treated_outcomes = outcomes[matched_pairs["treated_idx"]].values
        control_outcomes = outcomes[matched_pairs["control_idx"]].values

        # Calculate ATE
        ate = np.mean(treated_outcomes - control_outcomes)

        # Standard error and confidence interval
        differences = treated_outcomes - control_outcomes
        se = np.std(differences) / np.sqrt(len(differences))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # T-test
        t_stat, p_value = stats.ttest_rel(treated_outcomes, control_outcomes)

        results = {
            "ate": ate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_pairs": len(matched_pairs),
            "treated_mean": np.mean(treated_outcomes),
            "control_mean": np.mean(control_outcomes),
        }

        return results

    def check_balance(self, X, treatment, matched_pairs=None, threshold=0.1):
        """
        Check covariate balance after matching using Standardized Mean Difference

        SMD < 0.1 indicates good balance
        """
        if matched_pairs is None:
            matched_pairs = self.matched_pairs

        balance_stats = []

        for col_idx, col_name in enumerate(
            X.columns if hasattr(X, "columns") else range(X.shape[1])
        ):
            # Before matching
            treated_before = (
                X[treatment == 1].iloc[:, col_idx]
                if hasattr(X, "iloc")
                else X[treatment == 1, col_idx]
            )
            control_before = (
                X[treatment == 0].iloc[:, col_idx]
                if hasattr(X, "iloc")
                else X[treatment == 0, col_idx]
            )

            smd_before = self._calculate_smd(treated_before, control_before)

            # After matching
            treated_after = (
                X.iloc[matched_pairs["treated_idx"], col_idx]
                if hasattr(X, "iloc")
                else X[matched_pairs["treated_idx"], col_idx]
            )
            control_after = (
                X.iloc[matched_pairs["control_idx"], col_idx]
                if hasattr(X, "iloc")
                else X[matched_pairs["control_idx"], col_idx]
            )

            smd_after = self._calculate_smd(treated_after, control_after)

            balance_stats.append(
                {
                    "covariate": col_name,
                    "smd_before": smd_before,
                    "smd_after": smd_after,
                    "balanced": abs(smd_after) < threshold,
                }
            )

        balance_df = pd.DataFrame(balance_stats)

        print(f"\nBalance Check (SMD < {threshold} is good):")
        print(
            f"  Balanced covariates: {balance_df['balanced'].sum()}/{len(balance_df)}"
        )
        print(f"  Average SMD before: {balance_df['smd_before'].abs().mean():.3f}")
        print(f"  Average SMD after: {balance_df['smd_after'].abs().mean():.3f}")

        return balance_df

    @staticmethod
    def _calculate_smd(x1, x2):
        """Calculate Standardized Mean Difference"""
        mean1, mean2 = np.mean(x1), np.mean(x2)
        var1, var2 = np.var(x1), np.var(x2)
        pooled_std = np.sqrt((var1 + var2) / 2)

        if pooled_std == 0:
            return 0

        smd = (mean1 - mean2) / pooled_std
        return smd


class DifferenceInDifferences:
    def __init__(self):
        self.results = None

    def estimate(self, df, outcome_col, treatment_col, time_col, group_col):
        """
        Estimate DiD effect

        Args:
            df: DataFrame with panel data
            outcome_col: Name of outcome variable
            treatment_col: Binary treatment indicator (0=before, 1=after)
            time_col: Time period indicator
            group_col: Group indicator (0=control, 1=treated)
        """
        # Calculate means for each group-time combination
        group_time_means = df.groupby([group_col, treatment_col])[outcome_col].mean()

        # Extract values
        control_before = group_time_means.loc[0, 0]
        control_after = group_time_means.loc[0, 1]
        treated_before = group_time_means.loc[1, 0]
        treated_after = group_time_means.loc[1, 1]

        # Calculate DiD estimator
        control_change = control_after - control_before
        treated_change = treated_after - treated_before
        did_estimate = treated_change - control_change

        # Standard error using regression
        from statsmodels.formula.api import ols

        # Create interaction term
        df_copy = df.copy()
        df_copy["treatment_x_time"] = df_copy[treatment_col] * df_copy[group_col]

        # Run regression
        formula = f"{outcome_col} ~ {treatment_col} + {group_col} + treatment_x_time"
        model = ols(formula, data=df_copy).fit()

        did_coef = model.params["treatment_x_time"]
        did_se = model.bse["treatment_x_time"]
        did_pvalue = model.pvalues["treatment_x_time"]

        ci_lower = did_coef - 1.96 * did_se
        ci_upper = did_coef + 1.96 * did_se

        self.results = {
            "did_estimate": did_estimate,
            "se": did_se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": did_pvalue,
            "control_before": control_before,
            "control_after": control_after,
            "treated_before": treated_before,
            "treated_after": treated_after,
            "control_change": control_change,
            "treated_change": treated_change,
        }

        return self.results

    def test_parallel_trends(self, df, outcome_col, time_col, group_col, pre_periods):
        """
        Test parallel trends assumption using pre-treatment periods

        Args:
            pre_periods: List of pre-treatment time periods
        """
        from statsmodels.formula.api import ols

        # Filter to pre-treatment periods
        df_pre = df[df[time_col].isin(pre_periods)].copy()

        # Create time trend
        df_pre["time_numeric"] = pd.Categorical(df_pre[time_col]).codes
        df_pre["group_x_time"] = df_pre[group_col] * df_pre["time_numeric"]

        # Test if trends differ
        formula = f"{outcome_col} ~ time_numeric + {group_col} + group_x_time"
        model = ols(formula, data=df_pre).fit()

        # Interaction coefficient tests for differential trends
        coef = model.params["group_x_time"]
        pvalue = model.pvalues["group_x_time"]

        result = {
            "parallel_trends_hold": pvalue > 0.05,  # Fail to reject null = parallel
            "interaction_coef": coef,
            "p_value": pvalue,
        }

        print(f"\nParallel Trends Test:")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  P-value: {pvalue:.4f}")
        print(
            f"  Result: {'✓ Parallel trends supported' if result['parallel_trends_hold'] else '✗ Evidence against parallel trends'}"
        )

        return result


def calculate_uplift_curve(y_true, uplift_scores, n_bins=10):
    """
    Calculate uplift curve for model evaluation

    Similar to ROC curve but for uplift modeling
    """
    # Sort by uplift score (descending)
    sorted_indices = np.argsort(-uplift_scores)
    y_sorted = y_true[sorted_indices]

    # Calculate cumulative gains
    bin_size = len(y_true) // n_bins
    cumulative_uplift = []

    for i in range(1, n_bins + 1):
        top_n = i * bin_size
        top_uplift = y_sorted[:top_n].mean() - y_true.mean()
        cumulative_uplift.append(top_uplift)

    return np.array(cumulative_uplift)


def estimate_heterogeneous_effects(X, treatment, outcome, method="t_learner"):
    """
    Estimate Conditional Average Treatment Effects (CATE)

    Args:
        X: Covariates
        treatment: Binary treatment
        outcome: Outcome variable
        method: 't_learner', 's_learner', or 'x_learner'
    """
    from sklearn.ensemble import RandomForestRegressor

    if method == "t_learner":
        # Train separate models for treated and control
        model_t = RandomForestRegressor(n_estimators=100, random_state=42)
        model_c = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit models
        model_t.fit(X[treatment == 1], outcome[treatment == 1])
        model_c.fit(X[treatment == 0], outcome[treatment == 0])

        # Predict for everyone
        mu_t = model_t.predict(X)
        mu_c = model_c.predict(X)

        # CATE = difference
        cate = mu_t - mu_c

    elif method == "s_learner":
        # Single model with treatment as feature
        X_with_treatment = np.column_stack([X, treatment])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_with_treatment, outcome)

        # Predict under treatment and control
        X_t = np.column_stack([X, np.ones(len(X))])
        X_c = np.column_stack([X, np.zeros(len(X))])

        mu_t = model.predict(X_t)
        mu_c = model.predict(X_c)

        cate = mu_t - mu_c

    else:
        raise ValueError(f"Unknown method: {method}")

    return cate


def summarize_causal_results(results_dict, method_name):
    """
    Print formatted summary of causal inference results
    """
    print("\n" + "=" * 60)
    print(f"{method_name.upper()} RESULTS")
    print("=" * 60)

    if "ate" in results_dict:
        print(f"\nAverage Treatment Effect (ATE): {results_dict['ate']:.4f}")
        print(f"Standard Error: {results_dict['se']:.4f}")
        print(
            f"95% Confidence Interval: [{results_dict['ci_lower']:.4f}, {results_dict['ci_upper']:.4f}]"
        )
        print(f"P-value: {results_dict['p_value']:.4f}")

        if results_dict["p_value"] < 0.05:
            print("\n✓ Effect is statistically significant (p < 0.05)")
        else:
            print("\n✗ Effect is not statistically significant (p >= 0.05)")

    if "treated_mean" in results_dict and "control_mean" in results_dict:
        print(f"\nTreated group mean: {results_dict['treated_mean']:.4f}")
        print(f"Control group mean: {results_dict['control_mean']:.4f}")
        print(
            f"Difference: {results_dict['treated_mean'] - results_dict['control_mean']:.4f}"
        )

    if "n_pairs" in results_dict:
        print(f"\nSample size: {results_dict['n_pairs']} matched pairs")

    print("=" * 60)
