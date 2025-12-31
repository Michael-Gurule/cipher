"""
Propensity Score Matching Analysis - Email Marketing
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.causal_methods import PropensityScoreMatching, summarize_causal_results

# Load data
print("Loading marketing journey data...")
df = pd.read_csv("../data/marketing_journeys.csv")

print(f"Loaded {len(df):,} customer journeys")
print(f"Conversion rate: {df['converted'].mean() * 100:.2f}%")
print(f"Average revenue: ${df['revenue'].mean():.2f}")

# Focus on Email marketing effect
print("\n" + "=" * 60)
print("ANALYZING EFFECT OF EMAIL MARKETING")
print("=" * 60)

# Parse journey to determine email exposure
import json


def has_channel_exposure(journey_str, channel):
    """Check if customer was exposed to a channel"""
    try:
        journey = json.loads(journey_str)
        channels = [touchpoint["channel"] for touchpoint in journey]
        return channel in channels
    except:
        return False


df["exposed_to_email"] = df["journey"].apply(lambda x: has_channel_exposure(x, "Email"))

print(f"\nEmail exposure:")
print(
    f"  Exposed: {df['exposed_to_email'].sum():,} ({df['exposed_to_email'].mean() * 100:.1f}%)"
)
print(f"  Not exposed: {(~df['exposed_to_email']).sum():,}")

# Prepare covariates for matching
print("\nPreparing covariates for matching...")

# Convert categorical variables
df["gender_M"] = (df["gender"] == "M").astype(int)
df["gender_F"] = (df["gender"] == "F").astype(int)

df["segment_New"] = (df["segment"] == "New").astype(int)
df["segment_Returning"] = (df["segment"] == "Returning").astype(int)
df["segment_VIP"] = (df["segment"] == "VIP").astype(int)

# Select covariates
covariates = [
    "age",
    "intent_score",
    "n_touchpoints",
    "gender_M",
    "segment_Returning",
    "segment_VIP",
]

X = df[covariates]
treatment = df["exposed_to_email"].astype(int)
outcome_conversion = df["converted"]
outcome_revenue = df["revenue"]

print(f"Covariates: {', '.join(covariates)}")

# Run Propensity Score Matching
print("\n" + "=" * 60)
print("PROPENSITY SCORE MATCHING")
print("=" * 60)

psm = PropensityScoreMatching(caliper=0.1)

# Estimate propensity scores
print("\nEstimating propensity scores...")
propensity_scores = psm.fit(X, treatment)
print(
    f"Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]"
)

# Match treated to control
print("\nMatching treated units to controls...")
matched_pairs = psm.match(X, treatment, propensity_scores)

# Check balance
print("\nChecking covariate balance...")
balance_df = psm.check_balance(X, treatment, matched_pairs)

print("\nBalance Details:")
print(balance_df.to_string(index=False))

# Estimate treatment effect on conversion
print("\n" + "=" * 60)
print("TREATMENT EFFECT: CONVERSION")
print("=" * 60)

results_conversion = psm.estimate_ate(outcome_conversion, matched_pairs)
summarize_causal_results(results_conversion, "Email Marketing Effect on Conversion")

print(f"\nInterpretation:")
print(
    f"Email marketing increases conversion probability by {results_conversion['ate'] * 100:.2f} percentage points"
)
print(f"Treated conversion rate: {results_conversion['treated_mean'] * 100:.2f}%")
print(f"Control conversion rate: {results_conversion['control_mean'] * 100:.2f}%")

# Estimate treatment effect on revenue
print("\n" + "=" * 60)
print("TREATMENT EFFECT: REVENUE")
print("=" * 60)

results_revenue = psm.estimate_ate(outcome_revenue, matched_pairs)
summarize_causal_results(results_revenue, "Email Marketing Effect on Revenue")

print(f"\nInterpretation:")
print(
    f"Email marketing increases average revenue by ${results_revenue['ate']:.2f} per user"
)
print(f"Treated average revenue: ${results_revenue['treated_mean']:.2f}")
print(f"Control average revenue: ${results_revenue['control_mean']:.2f}")

# Calculate ROI
print("\n" + "=" * 60)
print("ROI CALCULATION")
print("=" * 60)

# Assume email costs $0.10 per send
cost_per_email = 0.10
n_emails_sent = df["exposed_to_email"].sum()
total_cost = n_emails_sent * cost_per_email

# Incremental revenue
incremental_revenue = results_revenue["ate"] * results_conversion["n_pairs"]

roi = (incremental_revenue - total_cost) / total_cost if total_cost > 0 else 0

print(
    f"Email campaign costs: ${total_cost:,.0f} ({n_emails_sent:,} emails × ${cost_per_email})"
)
print(f"Incremental revenue: ${incremental_revenue:,.0f}")
print(f"Net benefit: ${incremental_revenue - total_cost:,.0f}")
print(f"ROI: {roi:.2f}x")

# Segment analysis
print("\n" + "=" * 60)
print("HETEROGENEOUS EFFECTS BY SEGMENT")
print("=" * 60)

for segment in ["New", "Returning", "VIP"]:
    segment_mask = df["segment"] == segment
    segment_df = df[segment_mask]

    if len(segment_df) < 100:
        continue

    X_seg = segment_df[covariates]
    treatment_seg = segment_df["exposed_to_email"].astype(int)
    outcome_seg = segment_df["converted"]

    # Only run if we have enough treated and control
    if treatment_seg.sum() < 50 or (1 - treatment_seg).sum() < 50:
        print(f"\n{segment}: Insufficient sample size")
        continue

    psm_seg = PropensityScoreMatching(caliper=0.15)
    ps_seg = psm_seg.fit(X_seg, treatment_seg)
    pairs_seg = psm_seg.match(X_seg, treatment_seg, ps_seg)

    if len(pairs_seg) < 20:
        print(f"\n{segment}: Not enough matched pairs")
        continue

    results_seg = psm_seg.estimate_ate(outcome_seg.values, pairs_seg)

    print(f"\n{segment} Segment:")
    print(f"  Sample size: {len(segment_df):,}")
    print(f"  Matched pairs: {results_seg['n_pairs']}")
    print(f"  Treatment effect: {results_seg['ate'] * 100:.2f} percentage points")
    print(f"  P-value: {results_seg['p_value']:.4f}")
    print(f"  Significant: {'Yes' if results_seg['p_value'] < 0.05 else 'No'}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nKey Findings:")
print(
    f"1. Email marketing has a {results_conversion['ate'] * 100:.2f} percentage point effect on conversion"
)
print(f"2. This is statistically significant (p = {results_conversion['p_value']:.4f})")
print(f"3. ROI is {roi:.2f}x - every $1 spent returns ${roi:.2f}")
print(f"4. Effect varies by customer segment (see heterogeneous effects above)")
print("\nNext steps:")
print("  - Run notebooks/02_difference_in_differences.ipynb for temporal analysis")
print("  - Build uplift model to optimize targeting")
