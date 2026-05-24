import numpy as np
import pandas as pd
import networkx as nx

class CausalEngine:
    """
    Causal AI Engine representing the Structural Causal Model (SCM) of credit risk.
    Enables counterfactual queries (abduction, action, prediction) and recourse optimization.
    """
    def __init__(self):
        # Define the causal graph structure for NetworkX rendering
        self.graph = nx.DiGraph()
        
        # Add nodes with metadata (actionable, sensitive, immutable)
        self.nodes_meta = {
            'Age': {'type': 'immutable', 'label': 'Age'},
            'Gender': {'type': 'sensitive', 'label': 'Gender'},
            'Race': {'type': 'sensitive', 'label': 'Race'},
            'EmploymentLength': {'type': 'derived', 'label': 'Employment Length'},
            'AnnualIncome': {'type': 'actionable', 'label': 'Annual Income'},
            'MonthlyDebt': {'type': 'actionable', 'label': 'Monthly Debt'},
            'RequestedAmount': {'type': 'actionable', 'label': 'Requested Loan Amount'},
            'DTI': {'type': 'derived', 'label': 'Debt-to-Income (DTI)'},
            'CreditScore': {'type': 'derived', 'label': 'Credit Score'},
            'PriorDefaults': {'type': 'immutable', 'label': 'Prior Defaults'},
            'LoanToIncome': {'type': 'derived', 'label': 'Loan-to-Income'},
            'Approved': {'type': 'target', 'label': 'Loan Decision'}
        }
        
        for node, meta in self.nodes_meta.items():
            self.graph.add_node(node, **meta)
            
        # Add edges representing causal assumptions
        edges = [
            ('Age', 'EmploymentLength'),
            ('Age', 'AnnualIncome'),
            ('Age', 'CreditScore'),
            ('Gender', 'AnnualIncome'), # Systemic bias edge
            ('Race', 'AnnualIncome'),   # Systemic bias edge
            ('EmploymentLength', 'AnnualIncome'),
            ('EmploymentLength', 'CreditScore'),
            ('AnnualIncome', 'DTI'),
            ('AnnualIncome', 'LoanToIncome'),
            ('MonthlyDebt', 'DTI'),
            ('RequestedAmount', 'LoanToIncome'),
            ('DTI', 'CreditScore'),
            ('CreditScore', 'PriorDefaults'),
            # Edges into decision
            ('CreditScore', 'Approved'),
            ('DTI', 'Approved'),
            ('PriorDefaults', 'Approved'),
            ('LoanToIncome', 'Approved'),
            ('EmploymentLength', 'Approved'),
            ('Gender', 'Approved'), # Direct discrimination edge in historical data
            ('Race', 'Approved')     # Direct discrimination edge in historical data
        ]
        self.graph.add_edges_from(edges)

    def abduce_noise(self, profile):
        """
        Phase 1: Abduction. Computes the exogenous variables (noise terms) U for the observed profile.
        """
        age = profile['Age']
        gender = profile['Gender']
        race = profile['Race']
        emp_length = profile['EmploymentLength']
        income = profile['AnnualIncome']
        credit = profile['CreditScore']
        dti = profile['DTI']
        
        # U_emp = EmpLength - 0.4 * (Age - 18)
        u_emp = emp_length - 0.4 * max(0, age - 18)
        
        # U_income = Income - (30000 + 800 * (Age - 21) + 1500 * EmpLength - 6000 * Gender - 8000 * Race)
        expected_income = 30000 + 800 * (age - 21) + 1500 * emp_length - 6000 * gender - 8000 * race
        u_income = income - expected_income
        
        # U_credit = CreditScore - (550 + 2.5 * (Age - 21) + 3.5 * EmpLength - 150 * DTI)
        expected_credit = 550 + 2.5 * (age - 21) + 3.5 * emp_length - 150 * dti
        u_credit = credit - expected_credit
        
        return {
            'U_emp': u_emp,
            'U_income': u_income,
            'U_credit': u_credit
        }

    def propagate_intervention(self, profile, noise, interventions):
        """
        Phase 2 & 3: Action & Prediction. Propagates interventions on actionable features
        through the structural causal equations using the abduced noise terms.
        """
        cf = profile.copy()
        
        # Apply direct interventions (do-operations)
        for var, val in interventions.items():
            if var in self.nodes_meta and self.nodes_meta[var]['type'] == 'actionable':
                cf[var] = val
                
        # Propagate changes to downstream nodes in topological order
        # 1. Age, Gender, Race, EmploymentLength are unaffected by income/debt/loan interventions
        # 2. DTI = (MonthlyDebt * 12) / AnnualIncome
        cf['DTI'] = (cf['MonthlyDebt'] * 12) / cf['AnnualIncome']
        
        # 3. LoanToIncome = RequestedAmount / AnnualIncome
        cf['LoanToIncome'] = cf['RequestedAmount'] / cf['AnnualIncome']
        
        # 4. Credit Score = clip(550 + 2.5 * (Age - 21) + 3.5 * EmpLength - 150 * DTI + U_credit, 300, 850)
        expected_credit = 550 + 2.5 * (cf['Age'] - 21) + 3.5 * cf['EmploymentLength'] - 150 * cf['DTI']
        cf['CreditScore'] = int(np.clip(expected_credit + noise['U_credit'], 300, 850))
        
        # Note: PriorDefaults is kept constant here (past default is immutable)
        
        return cf

    def find_recourse_options(self, profile, model, feature_names, target_prob=0.6):
        """
        Finds distinct actionable recourse paths (counterfactuals) for a rejected applicant.
        We search across three actionable dimensions:
        1. Income Increase (up to +50%)
        2. Monthly Debt Reduction (up to 100% reduction)
        3. Requested Loan Amount Reduction (up to 60% reduction)
        
        Returns:
            list of dict: Recommended options with their costs, descriptions, and results.
        """
        noise = self.abduce_noise(profile)
        
        obs_income = profile['AnnualIncome']
        obs_debt = profile['MonthlyDebt']
        obs_amount = profile['RequestedAmount']
        
        options = []
        
        # Pre-calculate baseline prediction to check if already approved
        obs_features = pd.DataFrame([profile[feature_names]])
        if model.predict_proba(obs_features)[0, 1] >= target_prob:
            return [{'type': 'Already Approved', 'cost': 0, 'desc': 'Applicant is already approved.', 'diffs': {}}]
            
        # -------------------------------------------------------------
        # Path 1: Reduce Loan Amount Only (e.g. asking for less money)
        # -------------------------------------------------------------
        best_amt_recourse = None
        for amt_reduction in np.linspace(0, obs_amount * 0.6, 25):
            new_amount = max(5000, obs_amount - amt_reduction)
            cf = self.propagate_intervention(profile, noise, {'RequestedAmount': new_amount})
            
            # Predict
            cf_features = pd.DataFrame([cf[feature_names]])
            prob = model.predict_proba(cf_features)[0, 1]
            
            if prob >= target_prob:
                cost = amt_reduction / obs_amount
                best_amt_recourse = {
                    'type': 'Reduce Loan Request',
                    'cost': cost,
                    'desc': f"Reduce your loan request by ${int(amt_reduction):,} (new loan amount: ${int(new_amount):,})",
                    'interventions': {'RequestedAmount': new_amount},
                    'cf_profile': cf,
                    'prob': prob
                }
                break # Grid is sorted by increasing reduction, so first match is minimal cost
                
        if best_amt_recourse:
            options.append(best_amt_recourse)
            
        # -------------------------------------------------------------
        # Path 2: Pay Off Monthly Debt Only (reducing monthly commitments)
        # -------------------------------------------------------------
        best_debt_recourse = None
        if obs_debt > 0:
            for debt_reduction in np.linspace(0, obs_debt, 25):
                new_debt = max(0, obs_debt - debt_reduction)
                cf = self.propagate_intervention(profile, noise, {'MonthlyDebt': new_debt})
                
                # Predict
                cf_features = pd.DataFrame([cf[feature_names]])
                prob = model.predict_proba(cf_features)[0, 1]
                
                if prob >= target_prob:
                    cost = 2.0 * (debt_reduction / obs_debt)
                    best_debt_recourse = {
                        'type': 'Pay Down Debt',
                        'cost': cost,
                        'desc': f"Reduce your monthly debt payments by ${int(debt_reduction):,}/mo (new monthly debt: ${int(new_debt):,}/mo). This causally improves your DTI and Credit Score.",
                        'interventions': {'MonthlyDebt': new_debt},
                        'cf_profile': cf,
                        'prob': prob
                    }
                    break
                    
        if best_debt_recourse:
            options.append(best_debt_recourse)
            
        # -------------------------------------------------------------
        # Path 3: Increase Income Only (increasing earning capacity)
        # -------------------------------------------------------------
        best_inc_recourse = None
        for inc_increase in np.linspace(0, obs_income * 0.5, 25):
            new_income = obs_income + inc_increase
            cf = self.propagate_intervention(profile, noise, {'AnnualIncome': new_income})
            
            # Predict
            cf_features = pd.DataFrame([cf[feature_names]])
            prob = model.predict_proba(cf_features)[0, 1]
            
            if prob >= target_prob:
                cost = 3.0 * (inc_increase / obs_income)
                best_inc_recourse = {
                    'type': 'Increase Income',
                    'cost': cost,
                    'desc': f"Increase annual income by ${int(inc_increase):,} (new annual income: ${int(new_income):,}). This causally reduces DTI and raises Credit Score by improving stability.",
                    'interventions': {'AnnualIncome': new_income},
                    'cf_profile': cf,
                    'prob': prob
                }
                break
                
        if best_inc_recourse:
            options.append(best_inc_recourse)
            
        # -------------------------------------------------------------
        # Path 4: Combined Optimal Recourse (Grid search over all 3)
        # -------------------------------------------------------------
        best_combined = None
        min_combined_cost = float('inf')
        
        # Grid search with coarser intervals for performance
        inc_grid = np.linspace(0, obs_income * 0.3, 6)
        debt_grid = np.linspace(0, obs_debt, 6) if obs_debt > 0 else [0]
        amt_grid = np.linspace(0, obs_amount * 0.4, 6)
        
        for inc_add in inc_grid:
            for debt_sub in debt_grid:
                for amt_sub in amt_grid:
                    # Skip the zero intervention case
                    if inc_add == 0 and debt_sub == 0 and amt_sub == 0:
                        continue
                        
                    intervs = {}
                    if inc_add > 0: intervs['AnnualIncome'] = obs_income + inc_add
                    if debt_sub > 0: intervs['MonthlyDebt'] = max(0, obs_debt - debt_sub)
                    if amt_sub > 0: intervs['RequestedAmount'] = max(5000, obs_amount - amt_sub)
                    
                    cf = self.propagate_intervention(profile, noise, intervs)
                    cf_features = pd.DataFrame([cf[feature_names]])
                    prob = model.predict_proba(cf_features)[0, 1]
                    
                    if prob >= target_prob:
                        # Cost calculation: weights are Income=3.0, Debt=2.0, Amount=1.0
                        c_inc = 3.0 * (inc_add / obs_income) ** 2
                        c_debt = 2.0 * (debt_sub / obs_debt) ** 2 if obs_debt > 0 else 0
                        c_amt = 1.0 * (amt_sub / obs_amount) ** 2
                        cost = c_inc + c_debt + c_amt
                        
                        if cost < min_combined_cost:
                            min_combined_cost = cost
                            
                            desc_parts = []
                            if inc_add > 0:
                                desc_parts.append(f"increase income by ${int(inc_add):,}/yr")
                            if debt_sub > 0:
                                desc_parts.append(f"reduce debt by ${int(debt_sub):,}/mo")
                            if amt_sub > 0:
                                desc_parts.append(f"reduce loan request by ${int(amt_sub):,}")
                                
                            desc = "Balanced Strategy: " + ", ".join(desc_parts) + "."
                            
                            best_combined = {
                                'type': 'Balanced Strategy',
                                'cost': cost,
                                'desc': desc,
                                'interventions': intervs,
                                'cf_profile': cf,
                                'prob': prob
                            }
                            
        if best_combined:
            # Only add the combined strategy if it's cheaper than the single strategies, or if we want to show a variety
            # We sort options by cost anyway
            options.append(best_combined)
            
        # Sort options by cost (ascending)
        options = sorted(options, key=lambda x: x['cost'])
        return options
