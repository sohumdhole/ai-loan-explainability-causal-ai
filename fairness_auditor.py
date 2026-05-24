import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def compute_group_metrics(df, sensitive_col, target_col, pred_col):
    """
    Computes statistical and performance metrics for each group defined by the sensitive attribute.
    """
    groups = df[sensitive_col].unique()
    metrics = {}
    
    for g in groups:
        sub_df = df[df[sensitive_col] == g]
        total = len(sub_df)
        
        # Ground truth positives & negatives
        actual_pos = len(sub_df[sub_df[target_col] == 1])
        actual_neg = len(sub_df[sub_df[target_col] == 0])
        
        # Predictions
        pred_pos = len(sub_df[sub_df[pred_col] == 1])
        selection_rate = pred_pos / total if total > 0 else 0
        
        # Confusion matrix elements
        tp = len(sub_df[(sub_df[target_col] == 1) & (sub_df[pred_col] == 1)])
        fp = len(sub_df[(sub_df[target_col] == 0) & (sub_df[pred_col] == 1)])
        fn = len(sub_df[(sub_df[target_col] == 1) & (sub_df[pred_col] == 0)])
        tn = len(sub_df[(sub_df[target_col] == 0) & (sub_df[pred_col] == 0)])
        
        tpr = tp / actual_pos if actual_pos > 0 else 0
        fpr = fp / actual_neg if actual_neg > 0 else 0
        
        accuracy = (tp + tn) / total if total > 0 else 0
        
        metrics[g] = {
            'total': total,
            'selection_rate': selection_rate,
            'tpr': tpr,
            'fpr': fpr,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
    return metrics

def audit_fairness(df, sensitive_col, target_col, pred_col, privileged_group=0, unprivileged_group=1):
    """
    Computes standard fairness metrics comparing a privileged and unprivileged group.
    """
    group_metrics = compute_group_metrics(df, sensitive_col, target_col, pred_col)
    
    m_priv = group_metrics.get(privileged_group, {'selection_rate': 0, 'tpr': 0, 'fpr': 0, 'accuracy': 0})
    m_unpriv = group_metrics.get(unprivileged_group, {'selection_rate': 0, 'tpr': 0, 'fpr': 0, 'accuracy': 0})
    
    sr_priv = m_priv['selection_rate']
    sr_unpriv = m_unpriv['selection_rate']
    
    tpr_priv = m_priv['tpr']
    tpr_unpriv = m_unpriv['tpr']
    
    fpr_priv = m_priv['fpr']
    fpr_unpriv = m_unpriv['fpr']
    
    # 1. Demographic Parity Ratio (Disparate Impact Ratio)
    di_ratio = sr_unpriv / sr_priv if sr_priv > 0 else 1.0
    
    # 2. Demographic Parity Difference
    dp_diff = abs(sr_priv - sr_unpriv)
    
    # 3. Equal Opportunity Difference
    eo_diff = abs(tpr_priv - tpr_unpriv)
    
    # 4. Equalized Odds (maximum of TPR difference and FPR difference)
    fpr_diff = abs(fpr_priv - fpr_unpriv)
    equalized_odds_diff = max(eo_diff, fpr_diff)
    
    return {
        'group_metrics': group_metrics,
        'disparate_impact_ratio': di_ratio,
        'demographic_parity_diff': dp_diff,
        'equal_opportunity_diff': eo_diff,
        'fpr_difference': fpr_diff,
        'equalized_odds_diff': equalized_odds_diff
    }

def generate_fairness_charts(audit_results_std, audit_results_fair, sensitive_label="Gender"):
    """
    Generates interactive Plotly bar charts comparing standard and fair model outcomes.
    """
    categories = ['Selection Rate\n(Demographic Parity)', 'True Positive Rate\n(Equal Opportunity)', 'False Positive Rate\n(FPR Balance)']
    
    # Extract rates
    sr_std_priv = audit_results_std['group_metrics'][0]['selection_rate']
    sr_std_unpriv = audit_results_std['group_metrics'][1]['selection_rate']
    tpr_std_priv = audit_results_std['group_metrics'][0]['tpr']
    tpr_std_unpriv = audit_results_std['group_metrics'][1]['tpr']
    fpr_std_priv = audit_results_std['group_metrics'][0]['fpr']
    fpr_std_unpriv = audit_results_std['group_metrics'][1]['fpr']
    
    sr_fair_priv = audit_results_fair['group_metrics'][0]['selection_rate']
    sr_fair_unpriv = audit_results_fair['group_metrics'][1]['selection_rate']
    tpr_fair_priv = audit_results_fair['group_metrics'][0]['tpr']
    tpr_fair_unpriv = audit_results_fair['group_metrics'][1]['tpr']
    fpr_fair_priv = audit_results_fair['group_metrics'][0]['fpr']
    fpr_fair_unpriv = audit_results_fair['group_metrics'][1]['fpr']
    
    # Create figures
    fig_selection = go.Figure()
    
    # Standard Model
    fig_selection.add_trace(go.Bar(
        x=[f'Privileged ({sensitive_label}=0)', f'Unprivileged ({sensitive_label}=1)'],
        y=[sr_std_priv * 100, sr_std_unpriv * 100],
        name='Standard Model',
        marker_color='#FF4B4B'
    ))
    
    # Fair Model
    fig_selection.add_trace(go.Bar(
        x=[f'Privileged ({sensitive_label}=0)', f'Unprivileged ({sensitive_label}=1)'],
        y=[sr_fair_priv * 100, sr_fair_unpriv * 100],
        name='Debiased Model',
        marker_color='#00C0F2'
    ))
    
    fig_selection.update_layout(
        title=f"Approval Rate (%) by {sensitive_label} (Demographic Parity Audit)",
        yaxis_title="Approval Rate (%)",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=55, b=40)
    )
    
    # TPR Chart
    fig_tpr = go.Figure()
    fig_tpr.add_trace(go.Bar(
        x=[f'Privileged ({sensitive_label}=0)', f'Unprivileged ({sensitive_label}=1)'],
        y=[tpr_std_priv * 100, tpr_std_unpriv * 100],
        name='Standard Model',
        marker_color='#FF4B4B'
    ))
    fig_tpr.add_trace(go.Bar(
        x=[f'Privileged ({sensitive_label}=0)', f'Unprivileged ({sensitive_label}=1)'],
        y=[tpr_fair_priv * 100, tpr_fair_unpriv * 100],
        name='Debiased Model',
        marker_color='#00C0F2'
    ))
    
    fig_tpr.update_layout(
        title=f"True Positive Rate (%) by {sensitive_label} (Equal Opportunity Audit)",
        yaxis_title="True Positive Rate (%)",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=55, b=40)
    )
    
    return fig_selection, fig_tpr
