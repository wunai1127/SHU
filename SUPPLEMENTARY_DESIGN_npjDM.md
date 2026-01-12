# Neuro-Symbolic AI for Heart Transplant Perfusion: Supplementary Design Document

## For Submission to npj Digital Medicine

---

# Part I: Evaluation Framework (验证策略设计)

## 1. Clinical Validation Strategy

### 1.1 Retrospective Validation Study Design

#### Data Sources
- **Primary Dataset**: OCS/XVIVO machine perfusion records from partner hospitals (2018-2025)
- **Expected Sample Size**: N = 300-500 donor hearts
- **Data Types**:
  - Time-series physiological data (lactate, pH, aortic pressure, coronary flow, temperature)
  - Clinical notes and intervention records
  - Transplant outcomes (primary graft dysfunction, 30-day mortality, 1-year survival)
  - Knowledge graph triples extracted from literature (`heart_tx_all_merged_v8.json`)

#### Ground Truth Annotation Protocol
- **Annotators**: 3 senior perfusionists with >10 years experience
- **Annotation Task**: Retrospective evaluation of intervention decisions
- **Labels**:
  - Intervention timing appropriateness (early/optimal/late)
  - Intervention type correctness (correct/acceptable/suboptimal/incorrect)
  - Outcome contribution score (1-5 Likert scale)
- **Inter-rater Reliability**: Cohen's Kappa ≥ 0.7 required

### 1.2 Baseline Comparisons

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Baseline 1** | Logistic Regression for lactate clearance prediction | Traditional statistical approach |
| **Baseline 2** | LSTM on time-series only (no text, no knowledge graph) | Single-modality deep learning |
| **Baseline 3** | Human Expert Panel (3-5 senior perfusionists) | Clinical gold standard |

#### Baseline 1: Statistical Model
```
Lactate_clearance ~ Initial_lactate + Perfusion_pressure +
                    Temperature + Donor_age + Ischemic_time
```
- Model: Logistic Regression with L2 regularization
- Features: Standard clinical variables only
- Evaluation: AUC-ROC, calibration plot

#### Baseline 2: Single-Modality Deep Learning
- Architecture: Bidirectional LSTM with attention
- Input: Time-series data only (no textual features, no KG embeddings)
- Purpose: Demonstrate value of multi-modal fusion

#### Baseline 3: Human Expert Comparison
- **Protocol**:
  1. Present 50 randomly selected historical cases to expert panel
  2. Experts provide intervention recommendations blinded to actual outcomes
  3. Compare AI vs. expert accuracy using final transplant outcome as gold standard
- **Metrics**:
  - Decision concordance rate
  - Time-to-recommendation
  - Outcome prediction accuracy

### 1.3 Evaluation Metrics

#### Predictive Performance
| Metric | Target | Description |
|--------|--------|-------------|
| MAE (Lactate) | < 0.5 mmol/L | Mean Absolute Error for lactate prediction |
| MAE (pH) | < 0.05 | Mean Absolute Error for pH prediction |
| AUC-ROC | > 0.85 | Discrimination for adverse outcome prediction |
| Brier Score | < 0.15 | Calibration quality |

#### Decision Quality
| Metric | Definition | Target |
|--------|------------|--------|
| Cohen's Kappa | AI-Expert agreement | κ ≥ 0.6 |
| Intervention Precision | Correct interventions / Total AI recommendations | > 80% |
| Critical Alert Sensitivity | Detected critical events / Total critical events | > 95% |

#### Intervention Effectiveness (Causal Inference)
- **Counterfactual Analysis**: Using the Causal Inference Module (Agent 3)
- **Question**: "If AI-recommended intervention was applied, would the outcome have improved?"
- **Method**:
  - Propensity Score Matching
  - Doubly Robust Estimation
  - Sensitivity analysis for unmeasured confounders

```python
# Pseudo-code for causal effect estimation
def estimate_intervention_effect(data, intervention, outcome):
    # Step 1: Propensity score model
    ps_model = LogisticRegression()
    ps_model.fit(data.covariates, intervention)
    propensity_scores = ps_model.predict_proba(data.covariates)[:, 1]

    # Step 2: Outcome model with IPTW
    weights = compute_iptw(propensity_scores, intervention)
    ate = weighted_mean(outcome[intervention==1], weights[intervention==1]) - \
          weighted_mean(outcome[intervention==0], weights[intervention==0])

    # Step 3: Sensitivity analysis
    sensitivity_bounds = e_value(ate, outcome_variance)

    return ate, sensitivity_bounds
```

### 1.4 Ablation Studies

| Experiment | Removed Component | Purpose |
|------------|-------------------|---------|
| Ablation 1 | Knowledge Graph | Prove value of structured medical knowledge |
| Ablation 2 | Causal Inference Module | Prove value of counterfactual reasoning |
| Ablation 3 | Multi-Agent Collaboration | Prove value of specialized agents |
| Ablation 4 | Neuro-Symbolic Rules | Prove necessity of safety constraints |

---

# Part II: Neuro-Symbolic Value Proposition (神经符号价值论证)

## 2. Why Neuro-Symbolic is Essential for Perfusion AI

### 2.1 The Problem with Pure Deep Learning in Perfusion

#### Challenge 1: Small Sample Size
- **Reality**: Heart transplant centers perform 50-200 cases/year
- **Problem**: Pure DL requires thousands of samples; overfitting is inevitable
- **Solution**: Symbolic logic (Prolog rules) encodes expert knowledge to compensate for data scarcity

```prolog
% Example: Expert rule for pressure intervention
recommend_pressure_increase(State) :-
    lactate(State, Lactate), Lactate > 4.0,
    lactate_trend(State, increasing),
    current_pressure(State, Pressure), Pressure < 70,
    not(contraindication(State, pressure_increase)).
```

#### Challenge 2: Safety-Critical Domain
- **Reality**: Incorrect recommendations can cause graft failure
- **Problem**: Neural networks may output physiologically impossible values
- **Solution**: Symbolic layer acts as Safety Guardrail

#### Challenge 3: Regulatory Requirements
- **Reality**: Medical AI requires explainability (FDA, NMPA guidelines)
- **Problem**: Black-box models cannot satisfy regulatory audit requirements
- **Solution**: Neuro-symbolic hybrid provides traceable reasoning chains

### 2.2 The Neuro-Symbolic Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURO-SYMBOLIC ENGINE                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  Neural Layer   │───▶│ Symbolic Layer  │───▶│   Output    │ │
│  │  (Pattern       │    │ (Rule-based     │    │ (Validated  │ │
│  │   Recognition)  │    │  Verification)  │    │  Decision)  │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│         │                       │                              │
│         ▼                       ▼                              │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ - Trend Detection│    │ - Safety Rules  │                   │
│  │ - Anomaly Score │    │ - Physiological │                   │
│  │ - Risk Embedding│    │   Constraints   │                   │
│  │ - Similarity    │    │ - Domain Logic  │                   │
│  └─────────────────┘    └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Knowledge Integration Strategy

| Knowledge Source | Encoding Method | Usage |
|-----------------|-----------------|-------|
| Clinical Guidelines | Prolog Rules | Hard constraints |
| Expert Heuristics | Weighted Rules | Soft constraints |
| Literature Evidence | Knowledge Graph | Context retrieval |
| Case Histories | Vector Embeddings | Similarity matching |

---

## 3. Safety & Constraints Mechanism (安全护栏机制)

### 3.1 Multi-Layer Safety Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    SAFETY GUARDRAIL SYSTEM                    │
├───────────────────────────────────────────────────────────────┤
│  Layer 1: INPUT VALIDATION                                    │
│  ├── Physiological range checks                               │
│  ├── Data completeness verification                           │
│  └── Sensor anomaly detection                                 │
├───────────────────────────────────────────────────────────────┤
│  Layer 2: REASONING CONSTRAINTS                               │
│  ├── Rule-based intervention boundaries                       │
│  ├── Contraindication checking                                │
│  └── Interaction conflict detection                           │
├───────────────────────────────────────────────────────────────┤
│  Layer 3: OUTPUT FILTERING                                    │
│  ├── Physiological plausibility check                         │
│  ├── Rate-of-change limits                                    │
│  └── Hallucination detection                                  │
├───────────────────────────────────────────────────────────────┤
│  Layer 4: HUMAN OVERSIGHT                                     │
│  ├── Confidence threshold alerts                              │
│  ├── Mandatory human confirmation for critical actions        │
│  └── Audit logging                                            │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 Safety Rules Database

#### Hard Constraints (Absolute Boundaries)
```yaml
safety_rules:
  pressure:
    min: 40  # mmHg - below causes inadequate perfusion
    max: 100 # mmHg - above risks vascular damage
    max_change_rate: 10  # mmHg per minute

  temperature:
    min: 4   # °C - hypothermic preservation range
    max: 37  # °C - normothermic range
    max_change_rate: 1  # °C per 10 minutes

  flow:
    min: 0.5  # L/min
    max: 2.5  # L/min

  lactate:
    critical_threshold: 8.0  # mmol/L - requires immediate attention
    warning_threshold: 4.0   # mmol/L - elevated concern
```

#### Soft Constraints (Guideline-Based)
```python
class InterventionValidator:
    """Validates AI recommendations against clinical guidelines"""

    def validate_pressure_recommendation(self, current_state, recommendation):
        violations = []

        # Check absolute bounds
        if recommendation.target_pressure < 40 or recommendation.target_pressure > 100:
            violations.append(SafetyViolation(
                level="CRITICAL",
                message=f"Pressure {recommendation.target_pressure} outside safe range [40-100]",
                action="BLOCK"
            ))

        # Check rate of change
        pressure_change = abs(recommendation.target_pressure - current_state.pressure)
        if pressure_change > 20:
            violations.append(SafetyViolation(
                level="WARNING",
                message=f"Large pressure change ({pressure_change} mmHg) recommended",
                action="REQUIRE_CONFIRMATION"
            ))

        # Check contraindications
        if current_state.has_vascular_damage and recommendation.target_pressure > 80:
            violations.append(SafetyViolation(
                level="WARNING",
                message="High pressure contraindicated with vascular damage",
                action="SUGGEST_ALTERNATIVE"
            ))

        return violations
```

### 3.3 Hallucination Detection

The system employs multiple strategies to detect and filter AI hallucinations:

```python
class HallucinationDetector:
    """Detects potential AI hallucinations in recommendations"""

    def __init__(self, knowledge_graph, rule_engine):
        self.kg = knowledge_graph
        self.rules = rule_engine

    def detect(self, recommendation, context):
        scores = {}

        # 1. Knowledge Graph Grounding Check
        # Verify recommendation entities exist in KG
        kg_grounding = self.kg.verify_entities(recommendation.entities)
        scores['kg_grounding'] = kg_grounding

        # 2. Physiological Plausibility Check
        # Use symbolic rules to verify physiological consistency
        plausibility = self.rules.check_physiological_consistency(
            recommendation,
            context.patient_state
        )
        scores['plausibility'] = plausibility

        # 3. Evidence Citation Verification
        # Check if cited evidence actually supports the recommendation
        citation_validity = self.verify_citations(recommendation.citations)
        scores['citation_validity'] = citation_validity

        # 4. Self-Consistency Check
        # Generate multiple reasoning paths and check agreement
        consistency = self.check_reasoning_consistency(recommendation, context)
        scores['consistency'] = consistency

        # Aggregate score
        hallucination_risk = 1 - np.mean(list(scores.values()))

        return {
            'risk_score': hallucination_risk,
            'component_scores': scores,
            'is_hallucination': hallucination_risk > 0.3
        }
```

### 3.4 Audit Trail and Explainability

Every recommendation includes a complete audit trail:

```json
{
  "recommendation_id": "REC-2025-001234",
  "timestamp": "2025-01-12T10:30:00Z",
  "recommendation": {
    "action": "increase_perfusion_pressure",
    "target_value": 75,
    "unit": "mmHg",
    "urgency": "moderate"
  },
  "reasoning_chain": [
    {
      "step": 1,
      "observation": "Lactate level 4.2 mmol/L, trending upward",
      "source": "time_series_analyzer"
    },
    {
      "step": 2,
      "inference": "Inadequate tissue perfusion indicated",
      "evidence": ["Guideline: China Heart Transplant Standards 2019", "Similar case N=23"],
      "confidence": 0.87
    },
    {
      "step": 3,
      "action_derivation": "Rule R-PRESSURE-001 triggered",
      "rule_text": "IF lactate > 4.0 AND trend = increasing AND pressure < 70 THEN recommend pressure increase",
      "source": "symbolic_engine"
    }
  ],
  "safety_checks": {
    "passed": ["pressure_bounds", "rate_limit", "contraindication_check"],
    "warnings": [],
    "blocked": []
  },
  "confidence_score": 0.85,
  "requires_human_confirmation": false
}
```

---

# Part III: Human-Computer Interaction & Trust (人机交互与信任度)

## 4. Clinician Dashboard Design (临床仪表盘设计)

### 4.1 Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERFUSION AI DASHBOARD                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐ │
│  │    REAL-TIME VITALS       │  │      AI RECOMMENDATIONS            │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐  │  │  ┌────────────────────────────┐   │ │
│  │  │ LAC │ │ pH  │ │FLOW │  │  │  │ ⚠️ ALERT: Consider pressure │   │ │
│  │  │ 4.2 │ │7.32 │ │1.2L │  │  │  │    increase to 75 mmHg      │   │ │
│  │  │ ↑   │ │ →   │ │ →   │  │  │  │                              │   │ │
│  │  └─────┘ └─────┘ └─────┘  │  │  │ Confidence: 85%              │   │ │
│  │                           │  │  │ [Accept] [Modify] [Reject]   │   │ │
│  │  ┌─────────────────────┐  │  │  └────────────────────────────┘   │ │
│  │  │    Trend Graph      │  │  │                                    │ │
│  │  │   📈 ~~~~~~~~~~     │  │  │  WHY THIS RECOMMENDATION?          │ │
│  │  │                     │  │  │  • Lactate ↑ 4.2 (threshold: 4.0)  │ │
│  │  └─────────────────────┘  │  │  • Similar cases (N=50): 92% had   │ │
│  └───────────────────────────┘  │    improved outcomes with this     │ │
│                                  │    intervention                    │ │
│  ┌───────────────────────────┐  │  • Guideline: 中国心脏移植供心保护  │ │
│  │   SIMILAR CASES           │  │    技术规范(2019) Section 3.2      │ │
│  │  ┌─────────────────────┐  │  └────────────────────────────────────┘ │
│  │  │ Case #127: Similar  │  │                                         │
│  │  │ presentation, press │  │  ┌────────────────────────────────────┐ │
│  │  │ increased → Lactate │  │  │   SAFETY VERIFICATION              │ │
│  │  │ normalized in 45min │  │  │   ✓ Within physiological bounds   │ │
│  │  └─────────────────────┘  │  │   ✓ No contraindications          │ │
│  │  [View more cases...]     │  │   ✓ Rate of change acceptable     │ │
│  └───────────────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Explainability Display Components

#### Component 1: Risk Prediction Explanation
```
┌──────────────────────────────────────────────────────────────┐
│  GRAFT DYSFUNCTION RISK: 23%                                 │
│  ═══════════════════░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░          │
│                                                              │
│  Contributing Factors:                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Lactate clearance rate     ████████░░  +8%          │    │
│  │ Ischemic time (6.2h)       ██████░░░░  +6%          │    │
│  │ Donor age (52y)            █████░░░░░  +5%          │    │
│  │ Current pH trend           ███░░░░░░░  +3%          │    │
│  │ Protective factors         ░░░░░░░░░░  -2%          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Baseline risk for similar donors: 15%                       │
└──────────────────────────────────────────────────────────────┘
```

#### Component 2: Evidence Panel
```
┌──────────────────────────────────────────────────────────────┐
│  SUPPORTING EVIDENCE                                         │
│                                                              │
│  📚 Literature Evidence:                                     │
│  ├── "Enhanced Donor Lung Viability During Prolonged        │
│  │    Ex Vivo Perfusion" (Transpl Int. 2025)               │
│  │    Relevance: High | Cited for: pressure management      │
│  │                                                          │
│  ├── "中国心脏移植供心获取与保护技术规范(2019)"             │
│  │    Section 3.2: Perfusion pressure targets               │
│  │    Recommendation: 60-80 mmHg for hypothermic perfusion │
│  │                                                          │
│  📊 Similar Cases:                                           │
│  ├── N = 50 cases with similar presentation                 │
│  ├── Intervention success rate: 88%                         │
│  └── Average time to lactate normalization: 52 min          │
│                                                              │
│  [Expand Details] [View Full References]                     │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Human-in-the-Loop Interaction Modes

#### Mode 1: Supervisory Control (Default)
- AI provides recommendations with explanations
- Human reviews and approves/modifies/rejects
- All critical interventions require human confirmation

#### Mode 2: Advisory Mode
- AI monitors and alerts only
- No automatic recommendations
- Used during training or uncertain situations

#### Mode 3: Collaborative Mode
- Human and AI jointly review cases
- AI highlights areas of concern
- Human provides domain knowledge feedback

### 4.4 Online Learning from Expert Feedback

```python
class ExpertFeedbackLearner:
    """Learns from expert corrections and feedback"""

    def __init__(self, model, feedback_buffer_size=100):
        self.model = model
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)
        self.rule_adjustment_history = []

    def record_feedback(self, recommendation_id, expert_action, expert_reasoning):
        """Record expert feedback on AI recommendation"""
        feedback = {
            'recommendation_id': recommendation_id,
            'ai_recommendation': self.get_recommendation(recommendation_id),
            'expert_action': expert_action,  # 'accept', 'modify', 'reject'
            'expert_modification': expert_reasoning.get('modification'),
            'expert_reasoning': expert_reasoning.get('explanation'),
            'timestamp': datetime.now()
        }
        self.feedback_buffer.append(feedback)

        # Trigger learning if buffer is full
        if len(self.feedback_buffer) >= self.feedback_buffer.maxsize:
            self.update_model()

    def update_model(self):
        """Update model based on accumulated feedback"""
        # 1. Analyze rejection patterns
        rejections = [f for f in self.feedback_buffer if f['expert_action'] == 'reject']
        if len(rejections) > 10:
            self.analyze_rejection_patterns(rejections)

        # 2. Learn from modifications
        modifications = [f for f in self.feedback_buffer if f['expert_action'] == 'modify']
        if modifications:
            self.learn_from_modifications(modifications)

        # 3. Adjust confidence thresholds
        self.adjust_confidence_thresholds()

        # 4. Update rule weights (for soft constraints)
        self.update_rule_weights()

    def analyze_rejection_patterns(self, rejections):
        """Identify systematic issues in AI recommendations"""
        # Cluster rejections by context
        contexts = [r['ai_recommendation']['context'] for r in rejections]
        patterns = self.cluster_contexts(contexts)

        for pattern in patterns:
            if pattern.frequency > 0.3:  # >30% of rejections share this pattern
                self.flag_for_review(pattern)
                self.adjust_rule_for_context(pattern)

    def learn_from_modifications(self, modifications):
        """Learn expert preference from modifications"""
        for mod in modifications:
            original = mod['ai_recommendation']
            expert_version = mod['expert_modification']

            # Fine-tune model on expert corrections
            self.model.fine_tune_on_example(
                input_context=original['context'],
                original_output=original['action'],
                corrected_output=expert_version,
                weight=0.1  # Small learning rate for stability
            )
```

### 4.5 Trust Calibration Mechanisms

#### Confidence Display Strategy
- **High confidence (>85%)**: Green indicator, single confirmation
- **Medium confidence (60-85%)**: Yellow indicator, detailed explanation required
- **Low confidence (<60%)**: Red indicator, mandatory expert review, AI in advisory-only mode

#### Trust Building Features
1. **Accuracy History**: Display running accuracy of past recommendations
2. **Uncertainty Quantification**: Show prediction intervals, not just point estimates
3. **Limitation Disclosure**: Explicitly state what the AI cannot do
4. **Failure Mode Alerts**: Warn when operating outside training distribution

---

# Part IV: Experimental Design & Validation Plan (实验设计与验证计划)

## 5. Dataset Construction

### 5.1 Data Sources and Scale

| Data Type | Source | Expected Volume | Format |
|-----------|--------|-----------------|--------|
| Machine Perfusion Records | Partner hospitals | N = 300-500 cases | Time-series CSV |
| Clinical Notes | EMR systems | ~10,000 notes | Structured text |
| Intervention Logs | Perfusion devices | ~5,000 events | Timestamped events |
| Literature Corpus | PubMed, Guidelines | ~50,000 documents | JSON (see `heart_tx_all_merged_v8.json`) |
| Expert Annotations | Senior perfusionists | 3 annotators × 500 cases | Labeled JSON |

### 5.2 Annotation Protocol

```yaml
annotation_schema:
  case_id: string
  annotator_id: string
  annotation_date: datetime

  intervention_evaluation:
    timing:
      options: [too_early, optimal, too_late, not_needed]
      confidence: 1-5
    appropriateness:
      options: [correct, acceptable, suboptimal, incorrect]
      confidence: 1-5
    outcome_contribution:
      scale: 1-5  # 1=harmful, 3=neutral, 5=very beneficial

  alternative_recommendation:
    suggested_action: string
    reasoning: text

  case_complexity:
    scale: 1-5
    complicating_factors: list[string]
```

### 5.3 Train/Validation/Test Split

| Split | Proportion | Purpose | Notes |
|-------|------------|---------|-------|
| Training | 60% | Model training | Stratified by outcome |
| Validation | 20% | Hyperparameter tuning | Temporal split (earlier cases) |
| Test | 20% | Final evaluation | Temporal split (most recent cases) |

**Temporal Split Rationale**: Test set contains most recent cases to simulate real-world deployment scenario.

## 6. Evaluation Metrics (Detailed)

### 6.1 Predictive Performance Metrics

```python
metrics_config = {
    'regression_metrics': {
        'lactate_prediction': {
            'MAE': {'target': 0.5, 'unit': 'mmol/L'},
            'RMSE': {'target': 0.7, 'unit': 'mmol/L'},
            'R2': {'target': 0.8}
        },
        'ph_prediction': {
            'MAE': {'target': 0.05},
            'RMSE': {'target': 0.07}
        }
    },
    'classification_metrics': {
        'graft_dysfunction': {
            'AUC_ROC': {'target': 0.85},
            'Sensitivity': {'target': 0.90, 'note': 'Prioritize recall'},
            'Specificity': {'target': 0.75},
            'Brier_Score': {'target': 0.15}
        }
    }
}
```

### 6.2 Decision Quality Metrics

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| Cohen's Kappa | κ = (p_o - p_e)/(1 - p_e) | ≥ 0.6 | Substantial agreement with experts |
| Intervention Precision | TP / (TP + FP) | > 80% | Few unnecessary interventions |
| Critical Alert Sensitivity | TP_critical / Total_critical | > 95% | Miss no critical events |
| Time to Detection | Mean(t_AI - t_event) | < 5 min | Early warning capability |

### 6.3 Causal Effect Estimation

**Primary Analysis**: Average Treatment Effect (ATE) of AI-recommended interventions

```python
def compute_causal_metrics(data, treatment, outcome, confounders):
    """
    Estimate causal effect of AI-recommended interventions
    """
    # Propensity Score Matching
    ps_model = LogisticRegression().fit(data[confounders], treatment)
    propensity = ps_model.predict_proba(data[confounders])[:, 1]

    # IPTW Estimator
    weights = np.where(treatment == 1, 1/propensity, 1/(1-propensity))
    ate_iptw = np.average(outcome[treatment==1], weights=weights[treatment==1]) - \
               np.average(outcome[treatment==0], weights=weights[treatment==0])

    # Doubly Robust Estimator
    outcome_model = GradientBoostingRegressor().fit(
        data[confounders + [treatment]], outcome
    )
    mu_1 = outcome_model.predict(data[confounders].assign(treatment=1))
    mu_0 = outcome_model.predict(data[confounders].assign(treatment=0))

    ate_dr = np.mean(
        treatment * (outcome - mu_1) / propensity + mu_1 -
        (1-treatment) * (outcome - mu_0) / (1-propensity) - mu_0
    )

    # E-value for sensitivity analysis
    e_value = compute_e_value(ate_dr, np.std(outcome))

    return {
        'ATE_IPTW': ate_iptw,
        'ATE_DR': ate_dr,
        'E_value': e_value,
        'CI_95': bootstrap_ci(data, treatment, outcome)
    }
```

## 7. Comparative Experiments

### 7.1 Main Comparison Study

| Model | Description | Expected Performance |
|-------|-------------|---------------------|
| Our Method | Full Neuro-Symbolic Multi-Agent | AUC > 0.85, κ > 0.6 |
| Baseline 1 | Logistic Regression | AUC ~ 0.70 |
| Baseline 2 | LSTM (time-series only) | AUC ~ 0.78 |
| Baseline 3 | Human Expert Panel | κ comparison baseline |

### 7.2 Ablation Study Design

| Ablation | Removed Component | Hypothesis |
|----------|-------------------|------------|
| -KG | Knowledge Graph | Performance drops due to loss of structured knowledge |
| -Causal | Causal Inference Module | Intervention effectiveness estimation degrades |
| -MultiAgent | Single agent instead of 4 | Complex cases handled worse |
| -NeuroSymbolic | Pure neural, no symbolic rules | Safety violations increase, calibration worsens |

### 7.3 Clinical Utility Analysis

**Decision Curve Analysis**: Quantify net benefit across threshold probabilities

```python
def decision_curve_analysis(y_true, y_pred_proba, thresholds=np.arange(0.01, 0.99, 0.01)):
    """
    Compute net benefit for each threshold
    """
    net_benefits = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)

        # Net benefit formula
        net_benefit = (tp/n) - (fp/n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)

    return thresholds, net_benefits
```

**Marginal Heart Salvage Estimation**: How many marginal donor hearts could be utilized with AI assistance?

```python
def estimate_marginal_heart_salvage(historical_data, ai_predictions):
    """
    Estimate additional transplants enabled by AI-guided perfusion
    """
    # Identify marginal hearts (initially declined or high-risk)
    marginal_hearts = historical_data[historical_data['initial_assessment'] == 'marginal']

    # Count those where AI correctly identified salvageability
    ai_identified_salvageable = sum(
        (ai_predictions[marginal_hearts.index] == 'salvageable') &
        (marginal_hearts['actual_outcome'] == 'successful_transplant')
    )

    # Estimate annual impact
    annual_marginal = len(marginal_hearts) / historical_data['years_covered']
    estimated_additional_transplants = annual_marginal * (
        ai_identified_salvageable / len(marginal_hearts)
    )

    return {
        'marginal_hearts_total': len(marginal_hearts),
        'ai_identified_salvageable': ai_identified_salvageable,
        'estimated_annual_additional_transplants': estimated_additional_transplants,
        'confidence_interval': bootstrap_ci(...)
    }
```

---

## 8. Summary: Key Contributions for npj Digital Medicine

### Scientific Contributions
1. **First Neuro-Symbolic AI for Heart Perfusion**: Novel architecture combining neural pattern recognition with symbolic safety constraints
2. **Causal Inference for Intervention Effectiveness**: Move beyond correlation to causation in perfusion decision support
3. **Multi-Agent Collaboration Framework**: Specialized agents for different aspects of perfusion monitoring

### Clinical Contributions
1. **Improved Safety**: Multi-layer guardrail system prevents dangerous AI hallucinations
2. **Enhanced Explainability**: Every recommendation traceable to evidence and reasoning
3. **Human-AI Collaboration**: Designed for perfusionist workflow, not replacement

### Methodological Contributions
1. **Rigorous Validation Framework**: Head-to-head comparison with human experts
2. **Causal Effect Estimation**: Counterfactual analysis of intervention effectiveness
3. **Online Learning**: System improves from expert feedback over time

---

*Document Version: 1.0*
*Last Updated: 2026-01-12*
*Prepared for: npj Digital Medicine Submission*
