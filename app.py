import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from model_pipeline import generate_synthetic_data, train_and_save_pipeline
from causal_engine import CausalEngine
from fairness_auditor import audit_fairness, generate_fairness_charts

# Page Configuration
st.set_page_config(
    page_title="AI Credit Decision, Explainability & Causal Recourse System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container styling */
    .reportview-container {
        background: #0d1117;
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 2rem;
    }
    
    .header-title {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #00C0F2, #9B59B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Metrics panel */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-approved {
        font-size: 2.2rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .metric-value-rejected {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ef4444;
    }
    
    /* Recourse Card */
    .recourse-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 5px solid #00C0F2;
        padding: 1.25rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .recourse-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    
    .recourse-desc {
        font-size: 0.9rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and data artifacts
@st.cache_resource
def load_system_artifacts():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(data_dir, "model_artifacts.pkl")
    
    # Train if not present
    if not os.path.exists(artifacts_path):
        with st.spinner("Initializing models and dataset. This will take a moment..."):
            train_and_save_pipeline()
            
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
        
    test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    raw_data = pd.read_csv(os.path.join(data_dir, "synthetic_loan_data.csv"))
    
    return artifacts, test_data, raw_data

# Load artifacts
artifacts, test_data, raw_data = load_system_artifacts()
model_std = artifacts['model_std']
model_fair = artifacts['model_fair']
explainer = artifacts['explainer']
feature_names = artifacts['features']
stats = artifacts['stats']

# Initialize Causal AI Engine
causal_engine = CausalEngine()

# Render Header Card
st.markdown("""
<div class="header-card">
    <div class="header-title">AI Credit Decision, Explainability & Causal Recourse System</div>
    <div class="header-subtitle">A research-grade platform demonstrating eXplainable AI (XAI), structural causal inference, and algorithmic fairness auditing in banking credit risk models.</div>
</div>
""", unsafe_allow_html=True)

# Preset profiles for quick testing
st.sidebar.markdown("### 👤 Select Benchmark Profile")
preset_options = {
    "Custom Profile": None,
    "John (Rejected - High Debt)": {
        'Gender': 0, 'Race': 0, 'Age': 32, 'EmploymentLength': 4,
        'AnnualIncome': 48000, 'MonthlyDebt': 2100, 'RequestedAmount': 35000,
        'CreditScore': 580, 'PriorDefaults': 0
    },
    "Sophia (Rejected - Low Credit & Defaults)": {
        'Gender': 1, 'Race': 1, 'Age': 28, 'EmploymentLength': 2,
        'AnnualIncome': 52000, 'MonthlyDebt': 850, 'RequestedAmount': 20000,
        'CreditScore': 510, 'PriorDefaults': 1
    },
    "Alice (Approved - Low Risk)": {
        'Gender': 1, 'Race': 0, 'Age': 45, 'EmploymentLength': 12,
        'AnnualIncome': 95000, 'MonthlyDebt': 1100, 'RequestedAmount': 40000,
        'CreditScore': 740, 'PriorDefaults': 0
    }
}

preset_choice = st.sidebar.selectbox("Load preset applicant data:", list(preset_options.keys()))

# Interactive sliders for custom profiles
st.sidebar.markdown("### ⚙️ Edit Applicant Attributes")

# Initialize session state for inputs to allow recourse application
if 'app_profile' not in st.session_state or preset_choice != "Custom Profile":
    if preset_choice != "Custom Profile" and preset_options[preset_choice] is not None:
        st.session_state.app_profile = preset_options[preset_choice].copy()
    else:
        # Default baseline custom profile
        st.session_state.app_profile = {
            'Gender': 0, 'Race': 0, 'Age': 35, 'EmploymentLength': 6,
            'AnnualIncome': 65000, 'MonthlyDebt': 1200, 'RequestedAmount': 25000,
            'CreditScore': 620, 'PriorDefaults': 0
        }

p = st.session_state.app_profile

# Build sidebar controls connected to session state
gender = st.sidebar.selectbox("Gender", ["Male (0)", "Female (1)"], index=int(p['Gender']))
race = st.sidebar.selectbox("Race / Ethnicity", ["White (0)", "Minority (1)"], index=int(p['Race']))
age = st.sidebar.slider("Age", 21, 70, int(p['Age']))
emp_length = st.sidebar.slider("Employment Length (Years)", 0, max(0, age - 18), int(p['EmploymentLength']))
income = st.sidebar.slider("Annual Income ($)", 15000, 250000, int(p['AnnualIncome']), step=1000)
debt = st.sidebar.slider("Monthly Debt ($)", 0, 8000, int(p['MonthlyDebt']), step=50)
loan_amount = st.sidebar.slider("Requested Loan Amount ($)", 5000, 150000, int(p['RequestedAmount']), step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, int(p['CreditScore']))
defaults = st.sidebar.selectbox("Prior Defaults in last 2 years", ["No (0)", "Yes (1)"], index=int(p['PriorDefaults']))

# Update state based on controls
gender_val = 0 if "Male" in gender else 1
race_val = 0 if "White" in race else 1
defaults_val = 0 if "No" in defaults else 1

p['Gender'] = gender_val
p['Race'] = race_val
p['Age'] = age
p['EmploymentLength'] = emp_length
p['AnnualIncome'] = income
p['MonthlyDebt'] = debt
p['RequestedAmount'] = loan_amount
p['CreditScore'] = credit_score
p['PriorDefaults'] = defaults_val

# Derived variables
p['DTI'] = (p['MonthlyDebt'] * 12) / p['AnnualIncome']
p['LoanToIncome'] = p['RequestedAmount'] / p['AnnualIncome']

# Form features dataframe for standard XGBoost prediction
input_df = pd.DataFrame([p])[feature_names]

# Create Tabs
tab_xai, tab_causal, tab_fairness = st.tabs([
    "🔍 Credit Decision & Explainability (XAI)", 
    "🌱 Causal AI Playground & Recourse", 
    "⚖️ Algorithmic Fairness & Auditing"
])

# -------------------------------------------------------------
# TAB 1: Decision & SHAP Explanations
# -------------------------------------------------------------
with tab_xai:
    # Model predictions
    prob_std = model_std.predict_proba(input_df)[0, 1]
    decision_std = "Approved" if prob_std >= 0.5 else "Rejected"
    
    col_dec, col_metric1, col_metric2, col_metric3 = st.columns(4)
    
    with col_dec:
        if decision_std == "Approved":
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Loan Decision</div>
                <div class="metric-value-approved">APPROVED</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Loan Decision</div>
                <div class="metric-value-rejected">REJECTED</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Approval Probability</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #f8fafc;">{prob_std:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_metric2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Debt-to-Income (DTI)</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #f8fafc;">{p['DTI']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_metric3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Credit Score</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #f8fafc;">{p['CreditScore']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_shap_plot, col_shap_desc = st.columns([3, 2])
    
    # Compute SHAP values
    shap_vals = explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    if len(shap_vals.shape) > 1:
        shap_vals = shap_vals[0]
        
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
        
    # Sort SHAP contributions
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': [p[f] for f in feature_names],
        'SHAP': shap_vals
    })
    shap_df['Abs_SHAP'] = shap_df['SHAP'].abs()
    shap_df = shap_df.sort_values(by='Abs_SHAP', ascending=True)
    
    with col_shap_plot:
        st.markdown("### 📊 Local Feature Contributions (SHAP Waterfall)")
        
        # Plotly horizontal bar chart for SHAP values
        fig_shap = go.Figure()
        
        colors = ['#10b981' if v > 0 else '#ef4444' for v in shap_df['SHAP']]
        
        fig_shap.add_trace(go.Bar(
            y=shap_df['Feature'],
            x=shap_df['SHAP'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f} (val: {val:,.1f})" if f not in ['PriorDefaults', 'Age', 'EmploymentLength', 'CreditScore'] 
                  else f"{v:+.3f} (val: {int(val)})" for f, v, val in zip(shap_df['Feature'], shap_df['SHAP'], shap_df['Value'])],
            textposition='auto',
        ))
        
        fig_shap.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="SHAP Value (Log-odds Contribution)",
            yaxis_title="Features",
            margin=dict(l=40, r=40, t=10, b=40),
            height=450
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
    with col_shap_desc:
        st.markdown("### 📝 Human-Readable Interpretation")
        
        # Sort factors by impact
        top_factors = shap_df.sort_values(by='Abs_SHAP', ascending=False)
        
        st.write("This section translates the mathematical SHAP values into an intuitive audit breakdown.")
        
        factor_list = []
        for _, row in top_factors.head(3).iterrows():
            feat = row['Feature']
            val = row['Value']
            shap_v = row['SHAP']
            
            direction = "positive" if shap_v > 0 else "negative"
            action_word = "increased" if shap_v > 0 else "reduced"
            
            if feat == 'CreditScore':
                desc = f"Your **Credit Score** of `{int(val)}` had a **{direction}** impact, which **{action_word}** the approval log-odds by `{abs(shap_v):.3f}`."
            elif feat == 'DTI':
                desc = f"Your **Debt-to-Income (DTI)** ratio of `{val:.1%}` had a **{direction}** impact, which **{action_word}** the approval log-odds by `{abs(shap_v):.3f}`."
            elif feat == 'PriorDefaults':
                val_str = "Yes" if val == 1 else "No"
                desc = f"Having **Prior Defaults** set to `{val_str}` had a **{direction}** impact, which **{action_word}** the approval log-odds by `{abs(shap_v):.3f}`."
            elif feat == 'AnnualIncome':
                desc = f"Your **Annual Income** of `${int(val):,}` had a **{direction}** impact, which **{action_word}** the approval log-odds by `{abs(shap_v):.3f}`."
            elif feat == 'RequestedAmount':
                desc = f"Your **Requested Loan Amount** of `${int(val):,}` had a **{direction}** impact, which **{action_word}** the approval log-odds by `{abs(shap_v):.3f}`."
            else:
                desc = f"Your **{feat}** of `{val:,.1f}` had a **{direction}** impact, which **{action_word}** the approval log-odds by `{abs(shap_v):.3f}`."
                
            factor_list.append(desc)
            
        for i, factor in enumerate(factor_list):
            st.markdown(f"**{i+1}.** {factor}")
            
        # Explaining expected value
        st.markdown(f"""
        > **Statistical Baseline (Expected Log-Odds)**: `{base_val:.3f}`  
        > **Your Total Log-Odds**: `{sum(shap_vals)+base_val:.3f}`  
        > **Conversion to Probability**: `1 / (1 + exp(-Total Log-Odds)) = {prob_std:.1%}`
        """)

# -------------------------------------------------------------
# TAB 2: Causal AI Playground & Recourse
# -------------------------------------------------------------
with tab_causal:
    st.markdown("### 🔀 Algorithmic Recourse under Causal Knowledge")
    st.write("Conventional recourse algorithms recommend independent edits (e.g. 'increase credit score while keeping DTI constant'). In reality, features are causally connected. Our **Causal Recourse Engine** respects these connections: changing your income *causally propagates* to lower your DTI and raise your Credit Score, making the recommendation physically consistent.")
    
    col_c_left, col_c_right = st.columns([1, 1])
    
    with col_c_left:
        st.markdown("#### 📐 Structural Causal Model (SCM)")
        
        # Generate NetworkX graph image
        fig_net, ax = plt.subplots(figsize=(6, 5), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Layout positioning
        pos = {
            'Age': (-1.5, 2.5),
            'Gender': (0, 2.5),
            'Race': (1.5, 2.5),
            
            'EmploymentLength': (-1.2, 1.5),
            'MonthlyDebt': (0, 1.5),
            'RequestedAmount': (1.2, 1.5),
            
            'AnnualIncome': (-0.6, 0.5),
            'DTI': (0.6, 0.5),
            'LoanToIncome': (1.8, 0.5),
            
            'CreditScore': (-0.3, -0.5),
            'PriorDefaults': (1.0, -0.5),
            'Approved': (0.3, -1.6)
        }
        
        # Color coding nodes based on type
        node_colors = []
        for n in causal_engine.graph.nodes:
            ntype = causal_engine.graph.nodes[n]['type']
            if ntype == 'actionable':
                node_colors.append('#00C0F2')  # Blue
            elif ntype == 'sensitive':
                node_colors.append('#E74C3C')  # Red
            elif ntype == 'immutable':
                node_colors.append('#7F8C8D')  # Gray
            elif ntype == 'derived':
                node_colors.append('#9B59B6')  # Purple
            else:
                node_colors.append('#10B981')  # Green/Target
                
        nx.draw_networkx_nodes(causal_engine.graph, pos, ax=ax, node_color=node_colors, node_size=750, alpha=0.9)
        nx.draw_networkx_labels(causal_engine.graph, pos, ax=ax, font_color='#ffffff', font_size=7, font_family='Outfit')
        nx.draw_networkx_edges(causal_engine.graph, pos, ax=ax, edge_color='#475569', width=1.2, arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')
        
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig_net)
        
        # Node Legend
        st.markdown("""
        <div style="display: flex; gap: 15px; justify-content: center; font-size: 0.8rem; margin-top: -10px;">
            <span><span style="color:#7F8C8D">⬤</span> Immutable</span>
            <span><span style="color:#E74C3C">⬤</span> Sensitive</span>
            <span><span style="color:#00C0F2">⬤</span> Actionable</span>
            <span><span style="color:#9B59B6">⬤</span> Derived</span>
            <span><span style="color:#10B981">⬤</span> Target</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_c_right:
        st.markdown("#### 🎯 Actionable Recourse Recommendations")
        
        if decision_std == "Approved":
            st.success("🎉 Applicant is already approved! No recourse required.")
        else:
            recourse_options = causal_engine.find_recourse_options(p, model_std, feature_names, target_prob=0.55)
            
            if not recourse_options:
                st.warning("⚠️ No simple recourse path found within reasonable bounds. The credit risk is too high.")
            else:
                st.write("Here are the lowest-cost adjustments on actionable features that would result in approval:")
                
                # Render each recourse option as a card
                for i, opt in enumerate(recourse_options):
                    st.markdown(f"""
                    <div class="recourse-card">
                        <div class="recourse-title">{opt['type']} (Cost Index: {opt['cost']:.2f})</div>
                        <div class="recourse-desc">{opt['desc']}</div>
                        <div style="font-size: 0.8rem; color: #10b981; margin-top: 5px;">Projected Approval Probability: {opt['prob']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create button to apply this recourse state
                    if st.button(f"Apply Option {i+1}", key=f"rec_btn_{i}"):
                        # Update inputs in session state
                        for k, v in opt['interventions'].items():
                            st.session_state.app_profile[k] = v
                        # Force refresh
                        st.rerun()

    st.markdown("---")
    st.markdown("#### 🧪 Interactive Causal 'What-If' Playground")
    st.write("Perform *interventions* below. We abduce the exogenous background factors of the current applicant, apply your interventions, and compute how features causally evolve under the Structural Causal Model.")
    
    col_play1, col_play2, col_play3 = st.columns(3)
    
    with col_play1:
        play_inc_add = st.number_input("Intervention: Increase Annual Income ($)", min_value=0, max_value=100000, value=0, step=5000)
    with col_play2:
        play_debt_sub = st.number_input("Intervention: Reduce Monthly Debt ($)", min_value=0, max_value=int(p['MonthlyDebt']), value=0, step=100)
    with col_play3:
        play_amt_sub = st.number_input("Intervention: Reduce Loan Amount ($)", min_value=0, max_value=int(p['RequestedAmount'] - 5000), value=0, step=5000)
        
    # Execute propagation
    noise = causal_engine.abduce_noise(p)
    interventions = {}
    if play_inc_add > 0:
        interventions['AnnualIncome'] = p['AnnualIncome'] + play_inc_add
    if play_debt_sub > 0:
        interventions['MonthlyDebt'] = max(0, p['MonthlyDebt'] - play_debt_sub)
    if play_amt_sub > 0:
        interventions['RequestedAmount'] = max(5000, p['RequestedAmount'] - play_amt_sub)
        
    cf_profile = causal_engine.propagate_intervention(p, noise, interventions)
    
    # Predict on counterfactual profile
    cf_df = pd.DataFrame([cf_profile])[feature_names]
    cf_prob = model_std.predict_proba(cf_df)[0, 1]
    cf_decision = "Approved" if cf_prob >= 0.5 else "Rejected"
    
    # Display comparison table
    compare_df = pd.DataFrame({
        'Attribute': ['Annual Income', 'Monthly Debt', 'Requested Loan', 'DTI Ratio', 'Credit Score', 'Approval Probability', 'Outcome'],
        'Observed Value': [
            f"${p['AnnualIncome']:,}", 
            f"${p['MonthlyDebt']:,}", 
            f"${p['RequestedAmount']:,}", 
            f"{p['DTI']:.1%}", 
            str(p['CreditScore']), 
            f"{prob_std:.1%}", 
            decision_std
        ],
        'Causal Counterfactual Value': [
            f"${cf_profile['AnnualIncome']:,}", 
            f"${cf_profile['MonthlyDebt']:,}", 
            f"${cf_profile['RequestedAmount']:,}", 
            f"{cf_profile['DTI']:.1%}", 
            f"{cf_profile['CreditScore']} (Causally Updated)", 
            f"{cf_prob:.1%}", 
            cf_decision
        ]
    })
    
    st.table(compare_df)

# -------------------------------------------------------------
# TAB 3: Fairness & Bias Auditing
# -------------------------------------------------------------
with tab_fairness:
    st.markdown("### ⚖️ Algorithmic Fairness & Regulatory Auditing")
    st.write("Financial regulators audit credit decision models under the **Equal Credit Opportunity Act (ECOA)**. Even if sensitive features are omitted, machine learning models pick up on proxies (e.g. income gaps) and exhibit disparate impact. Here, we audit the standard model and compare it with our **Fairness-Mitigated Model** (trained using demographic re-weighting).")
    
    # Audit both models on the test set using Gender as the sensitive attribute
    
    # Calculate predictions for fair model to audit
    test_data_feat = test_data[feature_names]
    test_data['pred_std'] = model_std.predict(test_data_feat)
    test_data['pred_fair'] = model_fair.predict(test_data_feat)
    
    audit_std = audit_fairness(test_data, 'Gender', 'Approved', 'pred_std', privileged_group=0, unprivileged_group=1)
    audit_fair = audit_fairness(test_data, 'Gender', 'Approved', 'pred_fair', privileged_group=0, unprivileged_group=1)
    
    col_fair_std, col_fair_mit = st.columns(2)
    
    with col_fair_std:
        st.markdown("<h4 style='color:#ef4444;'>Standard XGBoost Model</h4>", unsafe_allow_html=True)
        
        acc_val = stats['acc_std']
        di_ratio = audit_std['disparate_impact_ratio']
        dp_diff = audit_std['demographic_parity_diff']
        eo_diff = audit_std['equal_opportunity_diff']
        
        compliant = "✅ Compliant" if di_ratio >= 0.8 else "❌ Non-Compliant"
        compliant_color = "#10b981" if di_ratio >= 0.8 else "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #ef4444;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span>Accuracy: <b>{acc_val:.1%}</b></span>
                <span style="color:{compliant_color}; font-weight:600;">{compliant}</span>
            </div>
            <hr style="margin: 10px 0; border:0; border-top:1px solid rgba(255,255,255,0.08);"/>
            <div style="text-align:left; font-size:0.9rem;">
                • <b>Disparate Impact (80% Rule) Ratio</b>: <span style="color:{compliant_color}; font-weight:bold;">{di_ratio:.3f}</span><br/>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:12px;">(Ratio of Female to Male approval rates)</span><br/>
                • <b>Demographic Parity Diff</b>: <b>{dp_diff:.3f}</b><br/>
                • <b>Equal Opportunity Difference</b>: <b>{eo_diff:.3f}</b><br/>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:12px;">(Difference in TPR between Female and Male)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_fair_mit:
        st.markdown("<h4 style='color:#00C0F2;'>Debiased Model (Reweighted)</h4>", unsafe_allow_html=True)
        
        acc_val_f = stats['acc_fair']
        di_ratio_f = audit_fair['disparate_impact_ratio']
        dp_diff_f = audit_fair['demographic_parity_diff']
        eo_diff_f = audit_fair['equal_opportunity_diff']
        
        compliant_f = "✅ Compliant" if di_ratio_f >= 0.8 else "❌ Non-Compliant"
        compliant_color_f = "#10b981" if di_ratio_f >= 0.8 else "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #00C0F2;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span>Accuracy: <b>{acc_val_f:.1%}</b></span>
                <span style="color:{compliant_color_f}; font-weight:600;">{compliant_f}</span>
            </div>
            <hr style="margin: 10px 0; border:0; border-top:1px solid rgba(255,255,255,0.08);"/>
            <div style="text-align:left; font-size:0.9rem;">
                • <b>Disparate Impact (80% Rule) Ratio</b>: <span style="color:{compliant_color_f}; font-weight:bold;">{di_ratio_f:.3f}</span><br/>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:12px;">(Ratio of Female to Male approval rates)</span><br/>
                • <b>Demographic Parity Diff</b>: <b>{dp_diff_f:.3f}</b><br/>
                • <b>Equal Opportunity Difference</b>: <b>{eo_diff_f:.3f}</b><br/>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:12px;">(Difference in TPR between Female and Male)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### 📈 Visual Fairness Audit Comparisons")
    
    fig_sel, fig_tpr = generate_fairness_charts(audit_std, audit_fair, sensitive_label="Gender")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(fig_sel, use_container_width=True)
    with col_chart2:
        st.plotly_chart(fig_tpr, use_container_width=True)
        
    st.markdown("""
    > **Methodology Note**: The debiased model is trained using **Kamiran & Calders sample re-weighting**. By computing weights for different subgroups in the training data, we balance the target variables relative to the sensitive attribute. This breaks systemic correlation proxies *without* removing features or adding sensitive features directly into the model inputs, maintaining strict regulatory compliance (ECOA) while drastically reducing bias.
    """)
