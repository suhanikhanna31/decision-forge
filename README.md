# DecisionForge – Revenue & Risk-Aware Decision Engine

DecisionForge is a backend decision system that converts machine learning predictions into accountable business actions. Instead of acting directly on churn probability, it evaluates return on investment (ROI), cost, and security risk before deciding whether to intervene, do nothing, or flag behavior.

Most retention systems act directly on churn probability, overspend on incentives, ignore abuse and anomalous behavior, and lack explainability and auditability. This results in wasted retention spend and poor business decisions.

DecisionForge introduces a decision layer between prediction and action. Every action must be economically justified (ROI-positive), pass security and risk checks, and be explainable and auditable.

Decision Flow:

User Features  
↓  
ML Scoring Layer  
• Churn Prediction (Logistic Regression)  
• Anomaly Detection (Isolation Forest)  
↓  
Decision Engine  
• Expected Value Calculation  
• ROI Threshold Check  
• Security Gating  
↓  
Decision Output  
• INTERVENE / DO_NOTHING / FLAG  
↓  
Audit Log  

System Components:
- Churn Model: Predicts churn probability using historical user behavior
- Anomaly Model: Detects suspicious or abusive behavior using isolation-based anomaly detection
- ROI Calculator: Computes expected monetary value of an intervention
- Decision Engine: Converts ML scores and business constraints into actions
- Security Gate: Blocks or flags risky interactions before spending resources
- Audit Logger: Records every decision with inputs, rationale, and timestamp
- YAML Configuration: Enables client-specific thresholds, costs, and risk tolerance without code changes

Business Impact (Simulated Evaluation):
DecisionForge was evaluated against a baseline strategy that intervenes purely on churn probability without considering cost or risk. In offline simulation, DecisionForge achieved approximately 8–10% simulated net revenue uplift by avoiding unprofitable incentives, reducing unnecessary retention spend by roughly 20%, and blocking 10–20% of risky or abusive interactions. Retention performance was maintained or improved through selective, ROI-driven intervention rather than blanket incentives.

How Evaluation Works:
A baseline policy intervenes whenever churn probability exceeds a fixed threshold. DecisionForge intervenes only when expected value is positive and the action passes security checks. For each strategy, net revenue is computed as revenue generated minus incentive cost. Aggregate net revenue is compared across users to measure uplift.

Technology Stack:
Python, scikit-learn (Logistic Regression, Isolation Forest), modular backend architecture, YAML-based configuration, and lightweight evaluation scripts. No external services or dashboards are required.

How to Run:
Install dependencies using `pip3 install scikit-learn pandas pyyaml`. Run a single decision example with `python3 run_decision.py`. Run the evaluation comparing baseline versus DecisionForge using `python3 evaluate.py`.

Project Philosophy:
Predictions do not equal decisions. Not acting is sometimes the optimal action. Every decision must justify its cost. Business logic must be explainable, auditable, and configurable.

What This Is Not:
This is not a dashboard, not a Kaggle notebook, and not a pure ML demo. DecisionForge is a decision infrastructure system designed to mirror real enterprise and consulting-grade backend decision platforms.