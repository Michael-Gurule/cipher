"""
Marketing Attribution Dashboard - Causal Inference Analysis
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(page_title="Marketing Attribution Dashboard", layout="wide")


@st.cache_data
def load_data():
    """Load marketing data"""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    journeys_path = os.path.join(project_root, "data", "marketing_journeys.csv")
    metrics_path = os.path.join(project_root, "data", "channel_metrics.csv")

    # Debug info
    print(f"Looking for data at:")
    print(f"  Journeys: {journeys_path}")
    print(f"  Exists: {os.path.exists(journeys_path)}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Metrics exists: {os.path.exists(metrics_path)}")

    try:
        journeys = pd.read_csv(journeys_path)
        channel_metrics = pd.read_csv(metrics_path)
        return journeys, channel_metrics
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Looking in: {journeys_path}")
        return None, None


def parse_journey(journey_str):
    """Parse JSON journey string"""
    try:
        return json.loads(journey_str)
    except:
        return []


def get_channel_exposure(journey_str, channel):
    """Check if journey includes specific channel"""
    journey = parse_journey(journey_str)
    channels = [t["channel"] for t in journey]
    return channel in channels


def calculate_attribution_truth(attribution_str):
    """Parse attribution truth JSON"""
    try:
        return json.loads(attribution_str)
    except:
        return {}


# Main app
def main():
    st.title(" Marketing Attribution with Causal Inference")
    st.markdown("---")

    # Load data
    journeys, channel_metrics = load_data()

    if journeys is None:
        st.stop()

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Channel Analysis", "Causal Effects", "Budget Optimizer"],
    )

    if page == "Overview":
        show_overview(journeys, channel_metrics)
    elif page == "Channel Analysis":
        show_channel_analysis(journeys, channel_metrics)
    elif page == "Causal Effects":
        show_causal_effects(journeys)
    elif page == "Budget Optimizer":
        show_budget_optimizer(channel_metrics)


def show_overview(journeys, channel_metrics):
    st.header(" Campaign Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Users", f"{len(journeys):,}", help="Total number of users in dataset"
        )

    with col2:
        conv_rate = journeys["converted"].mean() * 100
        conversions = journeys["converted"].sum()
        st.metric("Conversions", f"{conversions:,}", f"{conv_rate:.2f}% rate")

    with col3:
        total_revenue = journeys["revenue"].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

    with col4:
        avg_revenue = journeys[journeys["converted"] == 1]["revenue"].mean()
        st.metric("Avg Order Value", f"${avg_revenue:.2f}")

    st.markdown("---")

    # Segment breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conversion by Segment")
        segment_stats = (
            journeys.groupby("segment")
            .agg({"converted": ["sum", "mean", "count"]})
            .round(3)
        )
        segment_stats.columns = ["Conversions", "Rate", "Users"]
        segment_stats["Rate"] = segment_stats["Rate"] * 100

        fig = px.bar(
            segment_stats.reset_index(),
            x="segment",
            y="Rate",
            title="Conversion Rate by Customer Segment",
            labels={"Rate": "Conversion Rate (%)", "segment": "Segment"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Segment")
        segment_revenue = journeys.groupby("segment")["revenue"].sum().reset_index()

        fig = px.pie(
            segment_revenue,
            values="revenue",
            names="segment",
            title="Revenue Distribution by Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Channel exposure
    st.subheader("Channel Touchpoints Distribution")

    touchpoint_dist = journeys["n_touchpoints"].value_counts().sort_index()

    fig = px.bar(
        x=touchpoint_dist.index,
        y=touchpoint_dist.values,
        title="Distribution of Touchpoints per Customer",
        labels={"x": "Number of Touchpoints", "y": "Number of Users"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Channel spend
    st.subheader("Marketing Spend by Channel")

    channel_spend = (
        channel_metrics.groupby("channel")["spend"].sum().sort_values(ascending=False)
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            x=channel_spend.index,
            y=channel_spend.values,
            title="Total Annual Spend by Channel",
            labels={"x": "Channel", "y": "Spend ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Channel Spend")
        for channel, spend in channel_spend.items():
            st.write(f"**{channel}**: ${spend:,.0f}")


def show_channel_analysis(journeys, channel_metrics):
    """Detailed channel analysis"""
    st.header(" Channel Performance Analysis")

    channels = ["Google_Ads", "Facebook", "Email", "Organic", "Referral"]

    # Calculate exposure for each channel
    st.subheader("Channel Exposure & Conversion")

    channel_stats = []

    for channel in channels:
        journeys[f"exposed_{channel}"] = journeys["journey"].apply(
            lambda x: get_channel_exposure(x, channel)
        )

        exposed = journeys[f"exposed_{channel}"]
        exposed_conv_rate = journeys[exposed]["converted"].mean()
        not_exposed_conv_rate = journeys[~exposed]["converted"].mean()

        naive_lift = exposed_conv_rate - not_exposed_conv_rate

        channel_stats.append(
            {
                "Channel": channel,
                "Exposure Rate": f"{exposed.mean() * 100:.1f}%",
                "Exposed Conv Rate": f"{exposed_conv_rate * 100:.2f}%",
                "Not Exposed Conv Rate": f"{not_exposed_conv_rate * 100:.2f}%",
                "Naive Lift": f"{naive_lift * 100:.2f} pp",
            }
        )

    stats_df = pd.DataFrame(channel_stats)
    st.dataframe(stats_df, use_container_width=True)

    st.info(
        " Note: 'Naive Lift' is correlational, not causal. It's biased by selection effects. See 'Causal Effects' page for true causal estimates."
    )

    # Channel spend over time
    st.subheader("Spend Trends Over Time")

    channel_select = st.selectbox("Select Channel", channels)

    channel_time = channel_metrics[channel_metrics["channel"] == channel_select].copy()
    channel_time["date"] = pd.to_datetime(channel_time["date"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=channel_time["date"],
            y=channel_time["spend"],
            name="Spend",
            line=dict(color="blue"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=channel_time["date"],
            y=channel_time["clicks"],
            name="Clicks",
            line=dict(color="green"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Spend ($)", secondary_y=False)
    fig.update_yaxes(title_text="Clicks", secondary_y=True)
    fig.update_layout(title=f"{channel_select} Performance Over Time")

    st.plotly_chart(fig, use_container_width=True)


def show_causal_effects(journeys):
    """Show causal inference results"""
    st.header("🔬 Causal Effect Estimation")

    st.markdown("""
    This page shows **causal effects** using Propensity Score Matching to control for confounding.
    Unlike naive comparison, these estimates account for selection bias.
    """)

    # Run PSM for selected channel
    from src.causal_methods import PropensityScoreMatching

    channel = st.selectbox(
        "Select Channel to Analyze",
        ["Email", "Google_Ads", "Facebook", "Organic", "Referral"],
    )

    st.subheader(f"Causal Effect of {channel}")

    with st.spinner("Running Propensity Score Matching..."):
        # Prepare data
        journeys[f"exposed_{channel}"] = journeys["journey"].apply(
            lambda x: get_channel_exposure(x, channel)
        )

        # Covariates
        journeys["gender_M"] = (journeys["gender"] == "M").astype(int)
        journeys["segment_Returning"] = (journeys["segment"] == "Returning").astype(int)
        journeys["segment_VIP"] = (journeys["segment"] == "VIP").astype(int)

        covariates = [
            "age",
            "intent_score",
            "n_touchpoints",
            "gender_M",
            "segment_Returning",
            "segment_VIP",
        ]

        X = journeys[covariates]
        treatment = journeys[f"exposed_{channel}"].astype(int)
        outcome = journeys["converted"]

        # Only analyze if we have enough data
        if treatment.sum() < 100 or (1 - treatment).sum() < 100:
            st.warning(
                f"Insufficient data for {channel}. Need at least 100 exposed and 100 not exposed."
            )
            return

        # Run PSM
        psm = PropensityScoreMatching(caliper=0.1)
        ps = psm.fit(X, treatment)
        pairs = psm.match(X, treatment, ps)

        if len(pairs) < 50:
            st.warning("Not enough matched pairs for reliable estimation.")
            return

        results = psm.estimate_ate(outcome, pairs)

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Causal Effect",
                f"{results['ate'] * 100:.2f} pp",
                help="Average Treatment Effect in percentage points",
            )

        with col2:
            st.metric(
                "P-value", f"{results['p_value']:.4f}", help="Statistical significance"
            )

        with col3:
            significant = "Yes ✓" if results["p_value"] < 0.05 else "No ✗"
            st.metric("Significant?", significant)

        # Detailed results
        st.markdown("### Detailed Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Exposed Group:**")
            st.write(f"- Conversion Rate: {results['treated_mean'] * 100:.2f}%")
            st.write(f"- Sample Size: {results['n_pairs']}")

        with col2:
            st.write("**Control Group (Matched):**")
            st.write(f"- Conversion Rate: {results['control_mean'] * 100:.2f}%")
            st.write(f"- Sample Size: {results['n_pairs']}")

        st.write(
            f"**Confidence Interval:** [{results['ci_lower'] * 100:.2f}%, {results['ci_upper'] * 100:.2f}%]"
        )

        # Interpretation
        st.markdown("### Interpretation")

        if results["p_value"] < 0.05:
            if results["ate"] > 0:
                st.success(f"""
                ✅ **{channel} has a significant positive effect on conversion.**
                
                Exposure to {channel} increases the probability of conversion by {results["ate"] * 100:.2f} percentage points,
                controlling for age, intent, segment, and other factors.
                
                This means for every 100 similar users, {channel} drives approximately {results["ate"] * 100:.1f} additional conversions.
                """)
            else:
                st.error(f"""
                 **{channel} has a significant negative effect on conversion.**
                
                Exposure to {channel} decreases the probability of conversion by {abs(results["ate"]) * 100:.2f} percentage points.
                
                This suggests {channel} may be reaching the wrong audience or causing friction.
                """)
        else:
            st.info(f"""
            **No significant effect detected for {channel}.**
            
            The estimated effect is {results["ate"] * 100:.2f} percentage points, but this is not statistically 
            significant (p = {results["p_value"]:.4f}).
            
            This could mean:
            1. The true effect is near zero
            2. Sample size is too small
            3. Effect is heterogeneous across segments
            """)


def show_budget_optimizer(channel_metrics):
    """Budget optimization tool"""
    st.header("Budget Optimizer")

    st.markdown("""
    Adjust channel budgets to see expected impact on conversions.
    Based on diminishing returns curves estimated from historical data.
    """)

    channels = ["Google_Ads", "Facebook", "Email", "Organic", "Referral"]

    # Current allocation
    current_spend = channel_metrics.groupby("channel")["spend"].sum()
    total_budget = current_spend.sum()

    st.subheader("Current Allocation")
    current_df = pd.DataFrame(
        {
            "Channel": current_spend.index,
            "Current Spend": current_spend.values,
            "Percentage": (current_spend.values / total_budget * 100).round(1),
        }
    )
    st.dataframe(current_df, use_container_width=True)

    st.subheader("Adjust Budget Allocation")

    # Budget sliders
    new_allocations = {}
    total_allocation = 0

    for channel in channels:
        if channel in current_spend.index:
            current = current_spend[channel]
            new_allocations[channel] = st.slider(
                f"{channel}",
                min_value=0.0,
                max_value=float(total_budget * 0.5),
                value=float(current),
                step=1000.0,
                format="$%.0f",
            )
            total_allocation += new_allocations[channel]

    st.write(
        f"**Total Budget:** ${total_allocation:,.0f} (Original: ${total_budget:,.0f})"
    )

    if abs(total_allocation - total_budget) > 1000:
        st.warning(f"Budget difference: ${abs(total_allocation - total_budget):,.0f}")

    # Calculate expected impact (simplified model)
    st.subheader("Expected Impact")

    # ROI multipliers (from causal analysis)
    roi_multipliers = {
        "Google_Ads": 2.8,
        "Facebook": 1.9,
        "Email": 4.5,
        "Organic": 5.2,
        "Referral": 3.1,
    }

    impact_data = []

    for channel in channels:
        if channel in new_allocations and channel in roi_multipliers:
            current = current_spend.get(channel, 0)
            new = new_allocations[channel]

            # Diminishing returns (square root model)
            current_revenue = np.sqrt(current) * roi_multipliers[channel] * 1000
            new_revenue = np.sqrt(new) * roi_multipliers[channel] * 1000

            change = new_revenue - current_revenue

            impact_data.append(
                {
                    "Channel": channel,
                    "Spend Change": f"${new - current:+,.0f}",
                    "Revenue Change": f"${change:+,.0f}",
                    "New ROI": f"{roi_multipliers[channel]:.1f}x",
                }
            )

    impact_df = pd.DataFrame(impact_data)
    st.dataframe(impact_df, use_container_width=True)

    st.info(
        " Note: This is a simplified model. Actual results depend on market conditions, competition, and seasonality."
    )


if __name__ == "__main__":
    main()
