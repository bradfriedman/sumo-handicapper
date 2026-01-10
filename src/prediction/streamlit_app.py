"""
Streamlit UI for Sumo Bout Predictions
Beautiful, reactive interface for predicting sumo wrestling outcomes
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.prediction.prediction_engine import (
    load_model, predict_bout, search_rikishi_by_name,
    get_rikishi_by_id, DB_CONFIG
)
from src.core.fantasy_points import get_rank_label
import pymysql
from difflib import SequenceMatcher
import json
from src.training.update_model import (
    load_training_state, get_latest_bout_in_db, update_model
)

# Preferences file path (stored in project root)
PREFERENCES_FILE = os.path.join(project_root, '.streamlit_preferences.json')

# Default values
DEFAULT_BASHO_ID = 630
DEFAULT_DAY = 10

def load_preferences():
    """Load basho ID and day preferences from file"""
    try:
        if os.path.exists(PREFERENCES_FILE):
            with open(PREFERENCES_FILE, 'r') as f:
                data = json.load(f)
                basho_id = int(data.get('basho_id', DEFAULT_BASHO_ID))
                day = int(data.get('day', DEFAULT_DAY))
                # Validate ranges
                if 491 <= basho_id <= 700 and 1 <= day <= 15:
                    return basho_id, day
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass

    # Return defaults if file doesn't exist or is invalid
    return DEFAULT_BASHO_ID, DEFAULT_DAY

def save_preferences(basho_id, day):
    """Save basho ID and day preferences to file"""
    try:
        data = {
            'basho_id': basho_id,
            'day': day
        }
        with open(PREFERENCES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except (IOError, OSError):
        # Silently fail if we can't write the file
        pass

# Page configuration
st.set_page_config(
    page_title="Sumo Bout Predictor",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .winner-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def get_model():
    """Load and cache the prediction model"""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Cache database queries for rikishi search
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_search_rikishi_by_name(name_input, basho_id):
    """Cached wrapper for search_rikishi_by_name"""
    return search_rikishi_by_name(name_input, basho_id)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_get_rikishi_by_id(rikishi_id, basho_id):
    """Cached wrapper for get_rikishi_by_id"""
    return get_rikishi_by_id(rikishi_id, basho_id)

def calculate_edit_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    s1 = s1.lower()
    s2 = s2.lower()

    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def get_best_name_match(rikishi, search_input):
    """Get the best matching name for a rikishi and its edit distance"""
    search_input_lower = search_input.lower()

    # Check both ring name and real name
    names_to_check = []
    if rikishi.get('ring_name'):
        names_to_check.append(rikishi['ring_name'])
    names_to_check.append(rikishi['real_name'])

    # Find the name with minimum edit distance
    min_distance = float('inf')
    for name in names_to_check:
        distance = calculate_edit_distance(search_input_lower, name.lower())
        if distance < min_distance:
            min_distance = distance

    return min_distance

def display_rikishi_selector(label, key_prefix, basho_id):
    """Display rikishi selector with name search"""
    name_input = st.text_input(
        f"Enter {label} name (partial match works)",
        key=f"{key_prefix}_name_input",
        placeholder="e.g., Hakuho, Asashoryu"
    )

    if name_input:
        results = cached_search_rikishi_by_name(name_input, basho_id)

        if not results:
            st.warning(f"No rikishi found matching '{name_input}'")
            return None, None, None, None

        # Sort results by edit distance (closest match first)
        results_with_distance = [
            (r, get_best_name_match(r, name_input)) for r in results
        ]
        results_with_distance.sort(key=lambda x: x[1])  # Sort by edit distance
        results = [r[0] for r in results_with_distance]  # Extract sorted rikishi

        # Create display options
        options = []
        for r in results:
            display_name = r.get('ring_name') or r['real_name']
            real_name_part = f" ({r['real_name']})" if r.get('ring_name') else ""
            rank_part = f" - {get_rank_label(r['rank'])}" if r.get('rank') else " - Not in basho"
            options.append(f"{display_name}{real_name_part}{rank_part}")

        selected_idx = st.selectbox(
            f"Select {label}",
            range(len(options)),
            format_func=lambda i: options[i],
            key=f"{key_prefix}_select"
        )

        selected = results[selected_idx]
        display_name = selected.get('ring_name') or selected['real_name']

        return selected['id'], selected.get('rank'), selected.get('dob'), display_name

    return None, None, None, None

def create_probability_chart(rikishi_a_name, rikishi_b_name, prob_a, prob_b):
    """Create a horizontal bar chart for win probabilities"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[rikishi_a_name, rikishi_b_name],
        x=[prob_a * 100, prob_b * 100],
        orientation='h',
        marker=dict(
            color=['#ff6b6b', '#4ecdc4'],
            line=dict(color='rgb(8,48,107)', width=1.5)
        ),
        text=[f"{prob_a:.1%}", f"{prob_b:.1%}"],
        textposition='auto',
    ))

    fig.update_layout(
        title="Win Probability",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(range=[0, 100])
    )

    return fig

def create_model_comparison_chart(individual_preds, rikishi_a_name):
    """Create a chart comparing individual model predictions"""
    models = list(individual_preds.keys())
    probs = [individual_preds[m] * 100 for m in models]

    # Calculate dynamic range to zoom in on differences
    min_prob = min(probs)
    max_prob = max(probs)
    range_size = max_prob - min_prob

    # Add padding (10% of range or at least 5% total range)
    padding = max(range_size * 0.1, 5)
    y_min = max(0, min_prob - padding)
    y_max = min(100, max_prob + padding)

    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=probs,
            marker_color=['#ff6b6b', '#4ecdc4', '#95e1d3'],
            text=[f"{p:.2f}%" for p in probs],  # Show 2 decimal places
            textposition='outside',
        )
    ])

    fig.update_layout(
        title=f"Model Predictions for {rikishi_a_name} Win Probability (Zoomed)",
        xaxis_title="Model",
        yaxis_title="Probability (%)",
        height=350,
        yaxis=dict(range=[y_min, y_max])
    )

    return fig

def display_prediction_results(result, rikishi_a_name, rikishi_b_name):
    """Display prediction results in a beautiful format"""

    # Winner announcement
    winner_name = rikishi_a_name if result['predicted_winner_id'] == result['rikishi_a_id'] else rikishi_b_name
    loser_name = rikishi_b_name if result['predicted_winner_id'] == result['rikishi_a_id'] else rikishi_a_name

    # Get fantasy points for display
    fp = result['fantasy_points']
    if fp['rikishi_a_expected'] is not None:
        winner_fp = fp['rikishi_a_expected'] if result['predicted_winner_id'] == result['rikishi_a_id'] else fp['rikishi_b_expected']
        loser_fp = fp['rikishi_b_expected'] if result['predicted_winner_id'] == result['rikishi_a_id'] else fp['rikishi_a_expected']
        fp_text = f'<br/><span style="font-size: 1rem;">Expected Fantasy Points: {winner_name} ({winner_fp:.2f}) vs {loser_name} ({loser_fp:.2f})</span>'
    else:
        fp_text = ""

    st.markdown(f"""
    <div class="winner-card">
        🏆 Predicted Winner: {winner_name}
        <br/>
        <span style="font-size: 1.2rem;">Confidence: {result['confidence']:.1%}</span>
        {fp_text}
    </div>
    """, unsafe_allow_html=True)

    # Win probability chart
    st.plotly_chart(
        create_probability_chart(
            rikishi_a_name,
            rikishi_b_name,
            result['rikishi_a_win_probability'],
            result['rikishi_b_win_probability']
        ),
        use_container_width=True
    )

    # Three column layout for key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=f"{rikishi_a_name} Win Probability",
            value=f"{result['rikishi_a_win_probability']:.1%}"
        )

    with col2:
        st.metric(
            label=f"{rikishi_b_name} Win Probability",
            value=f"{result['rikishi_b_win_probability']:.1%}"
        )

    with col3:
        h2h = result['head_to_head']
        h2h_total = h2h['rikishi_a_wins'] + h2h['rikishi_b_wins']
        h2h_text = f"{h2h['rikishi_a_wins']}-{h2h['rikishi_b_wins']}" if h2h_total > 0 else "First Meeting"
        st.metric(
            label="Head-to-Head",
            value=h2h_text
        )

    # Expandable sections
    with st.expander("📊 Detailed Statistics", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Features")
            for key, value in result['key_features'].items():
                if value != 'N/A':
                    st.write(f"**{key.replace('_', ' ').title()}:** {value:.3f}")

        with col2:
            st.subheader("Head-to-Head Record")
            h2h = result['head_to_head']
            st.write(f"**{rikishi_a_name} wins:** {h2h['rikishi_a_wins']}")
            st.write(f"**{rikishi_b_name} wins:** {h2h['rikishi_b_wins']}")
            if h2h['rikishi_a_wins'] + h2h['rikishi_b_wins'] > 0:
                st.write(f"**{rikishi_a_name} win rate:** {h2h['rikishi_a_win_rate']:.1%}")

    with st.expander("🤖 Individual Model Predictions"):
        st.plotly_chart(
            create_model_comparison_chart(
                result['individual_predictions'],
                rikishi_a_name
            ),
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Random Forest", f"{result['individual_predictions']['random_forest']:.1%}")
        with col2:
            st.metric("LightGBM", f"{result['individual_predictions']['lightgbm']:.1%}")
        with col3:
            st.metric("XGBoost", f"{result['individual_predictions']['xgboost']:.1%}")

    # Fantasy points
    fp = result['fantasy_points']
    if fp['rikishi_a_expected'] is not None:
        with st.expander("🎮 Fantasy League Points"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"{rikishi_a_name}")
                rank_label = get_rank_label(fp['rikishi_a_rank'])
                st.write(f"**Rank:** {rank_label}")
                st.write(f"**Expected Points:** {fp['rikishi_a_expected']:.2f}")
                st.write(f"**Max Points (if win):** {fp['rikishi_a_potential']}")

            with col2:
                st.subheader(f"{rikishi_b_name}")
                rank_label = get_rank_label(fp['rikishi_b_rank'])
                st.write(f"**Rank:** {rank_label}")
                st.write(f"**Expected Points:** {fp['rikishi_b_expected']:.2f}")
                st.write(f"**Max Points (if win):** {fp['rikishi_b_potential']}")

            st.info("""
            **Fantasy Scoring:**
            - Base: 2 points for any win
            - Upset bonus: Additional points when lower-ranked wrestler wins
            - Expected points = Win probability × Potential points
            """)

def main():
    # Load model first
    with st.spinner("Loading prediction model..."):
        model_package = get_model()

    # Header
    st.markdown('<h1 class="main-header">🥋 Sumo Bout Predictor</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">AI-powered predictions using ensemble machine learning ({model_package["accuracy"]*100:.1f}% accuracy)</p>', unsafe_allow_html=True)

    # Display model info in sidebar
    st.sidebar.header("📈 Model Information")

    # Display training metadata
    if model_package.get('last_trained_basho_id') and model_package.get('last_trained_day'):
        st.sidebar.markdown(f"**Last Trained**  \nBasho {model_package['last_trained_basho_id']}  \nDay {model_package['last_trained_day']}")

    st.sidebar.metric("Training Bouts", f"{model_package['num_training_bouts']:,}")
    st.sidebar.metric("Accuracy", f"{model_package['accuracy']*100:.1f}%")

    # Display training date if available
    if model_package.get('training_date'):
        from datetime import datetime
        try:
            training_date = datetime.fromisoformat(model_package['training_date'])
            st.sidebar.caption(f"Updated: {training_date.strftime('%Y-%m-%d %H:%M')}")
        except:
            st.sidebar.caption(f"Updated: {model_package['training_date']}")

    st.sidebar.success("📡 Live data enabled")
    st.sidebar.info("Fresh head-to-head and recent records queried in real-time")

    # Mode selection
    st.sidebar.header("🎯 Prediction Mode")
    mode = st.sidebar.radio(
        "Choose prediction mode:",
        ["Single Bout Prediction", "Fantasy Roster (6 Bouts)", "Batch Predictions (CSV)", "Update Model"]
    )

    if mode == "Single Bout Prediction":
        st.header("Single Bout Prediction")

        # Load saved preferences for defaults
        default_basho_id, default_day = load_preferences()

        # Bout details
        col1, col2 = st.columns(2)

        with col1:
            basho_id = st.number_input(
                "Basho ID",
                min_value=491,
                max_value=700,
                value=default_basho_id,
                help="Tournament ID (491-630 in historical data)"
            )

        with col2:
            day = st.number_input(
                "Day",
                min_value=1,
                max_value=15,
                value=default_day,
                help="Day of the tournament (1-15)"
            )

        st.divider()

        # Rikishi selection
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Rikishi A")
            rikishi_a_id, rank_a, dob_a, name_a = display_rikishi_selector(
                "Rikishi A", "rikishi_a", basho_id
            )

        with col2:
            st.subheader("👤 Rikishi B")
            rikishi_b_id, rank_b, dob_b, name_b = display_rikishi_selector(
                "Rikishi B", "rikishi_b", basho_id
            )

        st.divider()

        # Predict button
        if st.button("🔮 Predict Bout Outcome", type="primary", use_container_width=True):
            if rikishi_a_id and rikishi_b_id:
                with st.spinner("Making prediction..."):
                    result = predict_bout(
                        model_package,
                        rikishi_a_id,
                        rikishi_b_id,
                        basho_id,
                        day,
                        rank_a,
                        rank_b,
                        dob_a,
                        dob_b
                    )

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                    if 'suggestion' in result:
                        st.info(result['suggestion'])
                else:
                    st.success("✅ Prediction complete!")
                    display_prediction_results(result, name_a, name_b)
                    # Save the basho_id and day for next time
                    save_preferences(basho_id, day)
            else:
                st.warning("⚠️ Please select both rikishi to make a prediction")

    elif mode == "Fantasy Roster (6 Bouts)":
        st.header("🎮 Fantasy Roster Optimizer")
        st.info("""
        **Fantasy League Mode**: Enter your 6 rostered rikishi and their opponents for the day.
        Get predictions sorted by expected fantasy points to help you choose the best 4 to start.
        """)

        # Load saved preferences for defaults
        default_basho_id, default_day = load_preferences()

        # Bout details (same for all 6 bouts)
        col1, col2 = st.columns(2)
        with col1:
            basho_id = st.number_input(
                "Basho ID",
                min_value=491,
                max_value=700,
                value=default_basho_id,
                help="Tournament ID (491-630 in historical data)",
                key="fantasy_basho"
            )
        with col2:
            day = st.number_input(
                "Day",
                min_value=1,
                max_value=15,
                value=default_day,
                help="Day of the tournament (1-15)",
                key="fantasy_day"
            )

        st.divider()
        st.subheader("Enter Your 6 Roster Bouts")

        # Store bout data
        bouts_data = []

        # Create 6 bout selectors
        for i in range(6):
            st.subheader(f"🥋 Bout {i+1}")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Your Rikishi (A)**")
                rikishi_a_id, rank_a, dob_a, name_a = display_rikishi_selector(
                    f"Rikishi {i+1}A", f"fantasy_a_{i}", basho_id
                )

            with col2:
                st.write("**Opponent (B)**")
                rikishi_b_id, rank_b, dob_b, name_b = display_rikishi_selector(
                    f"Rikishi {i+1}B", f"fantasy_b_{i}", basho_id
                )

            if rikishi_a_id and rikishi_b_id:
                bouts_data.append({
                    'rikishi_a_id': rikishi_a_id,
                    'rikishi_b_id': rikishi_b_id,
                    'rank_a': rank_a,
                    'rank_b': rank_b,
                    'dob_a': dob_a,
                    'dob_b': dob_b,
                    'name_a': name_a,
                    'name_b': name_b
                })

            # Add spacing between bouts
            if i < 5:
                st.markdown("---")

        st.divider()

        # Predict button
        if st.button("🔮 Analyze Roster", type="primary", use_container_width=True):
            if len(bouts_data) < 6:
                st.warning(f"⚠️ Please select all 6 bouts. Currently have {len(bouts_data)} complete bout(s).")
            else:
                with st.spinner("Running predictions for all 6 bouts..."):
                    # Run all predictions (in parallel via list comprehension)
                    results = []
                    for bout in bouts_data:
                        result = predict_bout(
                            model_package,
                            bout['rikishi_a_id'],
                            bout['rikishi_b_id'],
                            basho_id,
                            day,
                            bout['rank_a'],
                            bout['rank_b'],
                            bout['dob_a'],
                            bout['dob_b']
                        )
                        result['name_a'] = bout['name_a']
                        result['name_b'] = bout['name_b']
                        results.append(result)

                # Save preferences
                save_preferences(basho_id, day)

                # Sort by Rikishi A's expected fantasy points (descending)
                sorted_results = sorted(
                    results,
                    key=lambda x: x['fantasy_points']['rikishi_a_expected'] if x['fantasy_points']['rikishi_a_expected'] is not None else 0,
                    reverse=True
                )

                # Display sorted summary
                st.success("✅ All predictions complete!")
                st.subheader("📊 Roster Summary - Sorted by Expected Fantasy Points")

                # Create summary table
                summary_data = []
                total_expected = 0

                for i, result in enumerate(sorted_results, 1):
                    fp = result['fantasy_points']
                    a_expected = fp['rikishi_a_expected'] if fp['rikishi_a_expected'] is not None else 0
                    b_expected = fp['rikishi_b_expected'] if fp['rikishi_b_expected'] is not None else 0
                    total_expected += a_expected

                    # Determine winner indicator
                    is_a_favored = result['predicted_winner_id'] == result['rikishi_a_id']
                    indicator = "✓" if is_a_favored else "✗"

                    summary_data.append({
                        'Rank': f"#{i}",
                        'Win?': indicator,
                        'Bout': f"{result['name_a']} ({a_expected:.2f}) vs {result['name_b']} ({b_expected:.2f})",
                        'Your Rikishi Expected': f"{a_expected:.2f}",
                        'Win Probability': f"{result['rikishi_a_win_probability']:.1%}"
                    })

                # Display table without PyArrow
                # Create markdown table
                table_header = "| Rank | Win? | Bout | Your Rikishi Expected | Win Probability |\n"
                table_header += "|------|------|------|----------------------|------------------|\n"

                table_rows = ""
                for row in summary_data:
                    table_rows += f"| {row['Rank']} | {row['Win?']} | {row['Bout']} | {row['Your Rikishi Expected']} | {row['Win Probability']} |\n"

                st.markdown(table_header + table_rows)

                # Calculate top 4 expected points
                top_4_expected = sum([
                    float(summary_data[i]['Your Rikishi Expected'])
                    for i in range(min(4, len(summary_data)))
                ]) if summary_data else 0

                # Total expected points (top 4)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 1.5rem; border-radius: 0.5rem;
                            text-align: center; font-size: 2rem; font-weight: bold; margin: 1rem 0;">
                    💰 Top 4 Expected Fantasy Points: {top_4_expected:.2f}
                </div>
                """, unsafe_allow_html=True)

                st.info("""
                **💡 Strategy Tip**: Consider starting the top 4 rikishi by expected points.
                The "Win?" column shows ✓ if your rikishi is favored to win.
                """)

                # Detailed results for each bout
                st.divider()
                st.subheader("📋 Detailed Bout Analysis")

                for i, result in enumerate(sorted_results, 1):
                    with st.expander(f"Bout #{i}: {result['name_a']} vs {result['name_b']}", expanded=False):
                        if 'error' in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            display_prediction_results(result, result['name_a'], result['name_b'])

    elif mode == "Batch Predictions (CSV)":
        st.header("Batch Predictions from CSV")

        st.info("""
        **CSV Format Requirements:**
        - Required columns: `rikishi_a_id`, `rikishi_b_id`, `basho_id`, `day`
        - Optional columns: `rikishi_a_rank`, `rikishi_b_rank`, `rikishi_a_dob`, `rikishi_b_dob`
        """)

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} bouts")
                st.dataframe(df.head())

                # Validate columns
                required = ['rikishi_a_id', 'rikishi_b_id', 'basho_id', 'day']
                missing = [col for col in required if col not in df.columns]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    if st.button("🚀 Run Batch Predictions", type="primary"):
                        progress_bar = st.progress(0)
                        results = []

                        for idx, row in df.iterrows():
                            result = predict_bout(
                                model_package,
                                row['rikishi_a_id'],
                                row['rikishi_b_id'],
                                row['basho_id'],
                                row['day'],
                                row.get('rikishi_a_rank'),
                                row.get('rikishi_b_rank'),
                                row.get('rikishi_a_dob'),
                                row.get('rikishi_b_dob')
                            )
                            results.append(result)
                            progress_bar.progress((idx + 1) / len(df))

                        st.success(f"✅ Completed {len(results)} predictions!")

                        # Display results
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="predictions_results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

    elif mode == "Update Model":
        st.header("🔧 Model Update")
        st.info("""
        **Incremental Model Update**: Update the prediction model with new bout data without retraining from scratch.
        The system tracks the most recent training state and only processes new data when available.
        """)

        # Load current training state
        state = load_training_state()

        st.subheader("📊 Current Model Status")

        if state['last_trained_basho_id']:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Last Trained Basho", state['last_trained_basho_id'])

            with col2:
                st.metric("Last Trained Day", state['last_trained_day'])

            with col3:
                st.metric("Training Bouts", f"{state['num_training_bouts']:,}")

            with col4:
                if state['accuracy']:
                    st.metric("Model Accuracy", f"{state['accuracy']*100:.2f}%")
                else:
                    st.metric("Model Accuracy", "N/A")

            if state['last_training_date']:
                from datetime import datetime
                try:
                    training_date = datetime.fromisoformat(state['last_training_date'])
                    st.caption(f"Last updated: {training_date.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.caption(f"Last updated: {state['last_training_date']}")
        else:
            st.warning("⚠️ No previous training found. Model will be trained from scratch.")

        st.divider()

        # Check for new data
        st.subheader("🔍 Check for New Data")

        if st.button("🔎 Check Database", use_container_width=True):
            with st.spinner("Querying database for latest bout..."):
                latest_basho, latest_day = get_latest_bout_in_db()

                if latest_basho is None:
                    st.error("❌ Could not connect to database or no bouts found")
                else:
                    st.success(f"✅ Latest bout in database: Basho {latest_basho}, Day {latest_day}")

                    # Check if update is needed
                    if (state['last_trained_basho_id'] == latest_basho and
                        state['last_trained_day'] == latest_day):
                        st.info("✓ Model is already up-to-date! No new bouts to train on.")
                    else:
                        st.warning(f"⚡ New data available! Model can be updated.")
                        if state['last_trained_basho_id']:
                            st.write(f"Current: Basho {state['last_trained_basho_id']}, Day {state['last_trained_day']}")
                            st.write(f"Latest: Basho {latest_basho}, Day {latest_day}")

        st.divider()

        # Update model section
        st.subheader("🚀 Update Model")

        use_full_corpus = st.checkbox(
            "Train on FULL CORPUS (all historical bouts)",
            value=True,
            help="Recommended: Train on all available data for best accuracy. Uncheck to train on recent data only (faster but less accurate)."
        )

        if use_full_corpus:
            st.info("✓ Will train on ALL bouts in the database (recommended for production)")
        else:
            st.warning("⚠️ Will train on recent data only (50 bashos). Use this for testing only.")

        if st.button("🔥 Update Model with New Data", type="primary", use_container_width=True):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Checking for new data...")
                progress_bar.progress(10)

                # Run the update
                status_text.text("Training model... This may take several minutes.")
                progress_bar.progress(20)

                # Note: The update_model function doesn't provide incremental progress,
                # so we'll show an indeterminate state during training
                result = update_model(use_full_corpus=use_full_corpus, verbose=False)

                progress_bar.progress(100)
                status_text.text("Update complete!")

                if result is None:
                    st.error("❌ Model update failed. Check logs for details.")
                elif result['updated']:
                    st.success("✅ Model successfully updated!")

                    # Show before/after comparison
                    st.subheader("📈 Update Summary")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Before Update**")
                        if state['last_trained_basho_id']:
                            st.write(f"Basho: {state['last_trained_basho_id']}, Day: {state['last_trained_day']}")
                            st.write(f"Bouts: {state['num_training_bouts']:,}")
                            if state['accuracy']:
                                st.write(f"Accuracy: {state['accuracy']*100:.2f}%")
                        else:
                            st.write("No previous training")

                    with col2:
                        st.markdown("**After Update**")
                        st.write(f"Basho: {result['last_basho']}, Day: {result['last_day']}")
                        st.write(f"Bouts: {result['num_bouts']:,}")
                        st.write(f"Accuracy: {result['accuracy']*100:.2f}%")

                    # Calculate improvement
                    if state['accuracy'] and result['accuracy']:
                        improvement = (result['accuracy'] - state['accuracy']) * 100
                        if improvement > 0:
                            st.success(f"📊 Accuracy improved by {improvement:.2f} percentage points!")
                        elif improvement < 0:
                            st.warning(f"📊 Accuracy decreased by {abs(improvement):.2f} percentage points.")
                        else:
                            st.info("📊 Accuracy remained the same.")

                    st.info("💡 **Next Step**: Reload this page to use the updated model for predictions.")

                else:
                    st.info("✓ Model is already up-to-date. No new bouts to train on.")

            except Exception as e:
                progress_bar.progress(0)
                status_text.text("")
                st.error(f"❌ Error during model update: {str(e)}")
                st.write("Please check that all dependencies are installed and the database is accessible.")

    # Footer
    st.divider()

    # Build footer text with available info
    footer_parts = [
        "Powered by ensemble machine learning (Random Forest, LightGBM, XGBoost)",
        f"Model trained on {model_package['num_training_bouts']:,} historical bouts | {model_package['accuracy']*100:.1f}% accuracy"
    ]

    if model_package.get('last_trained_basho_id') and model_package.get('last_trained_day'):
        footer_parts.append(f"Last trained: Basho {model_package['last_trained_basho_id']}, Day {model_package['last_trained_day']}")

    footer_html = "<br/>".join(f"<p>{part}</p>" for part in footer_parts)

    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        {footer_html}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
