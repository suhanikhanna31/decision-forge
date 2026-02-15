"""
AI-powered decision explainer using Ollama (FREE local AI).
"""

from typing import Dict, Any, List
import ollama


class AIExplainer:
    """
    Uses Ollama local AI to generate explanations for decisions.
    """
    
    def __init__(self, model: str = "llama3.2"):
        """Initialize the AI explainer with Ollama model."""
        self.model = model
        # Test connection
        try:
            ollama.list()
            print(f"✓ Connected to Ollama with model: {model}")
        except Exception as e:
            print(f"⚠️  Ollama not running: {e}")
            print("Run: ollama serve")
    
    def explain_decision(
        self,
        decision: Dict[str, Any],
        inputs: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> str:
        """Generate a natural language explanation for a decision."""
        prompt = self._build_explanation_prompt(decision, inputs, user_context)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    def explain_intervention_recommendation(
        self,
        decision: Dict[str, Any],
        inputs: Dict[str, Any],
        expected_value: float
    ) -> Dict[str, str]:
        """Generate detailed recommendations for intervention actions."""
        prompt = f"""You are an AI advisor for a customer retention system. Based on the following analysis, provide actionable recommendations:

Decision: {decision['decision']}
Reason: {decision['reason']}
Expected Value: ${expected_value:.2f}

User Metrics:
- Anomaly Score: {inputs.get('anomaly_score', 'N/A')}
- Request Count Today: {inputs.get('request_count_today', 'N/A')}
- Churn Probability: {inputs.get('churn', 0) * 100:.1f}%

Provide a response in this format:
SUMMARY: [One-line summary]
RATIONALE: [Why this decision maximizes value]
SUGGESTED_ACTIONS: [3-4 specific actions to take]"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response['message']['content']
            return self._parse_structured_response(response_text)
        except Exception as e:
            return {
                'summary': f'Error: {e}',
                'rationale': '',
                'suggested_actions': ''
            }
    
    def generate_personalized_message(
        self,
        customer_profile: Dict[str, Any],
        intervention_type: str,
        offer_details: Dict[str, Any] = None
    ) -> str:
        """Generate a personalized customer message for interventions."""
        prompt = f"""Generate a friendly, personalized customer retention message.

Customer Profile:
- Tenure: {customer_profile.get('tenure', 'N/A')} months
- Monthly Charges: ${customer_profile.get('monthly_charges', 'N/A')}

Intervention Type: {intervention_type}
Offer: {offer_details if offer_details else 'Loyalty appreciation'}

Write a warm, concise message (2-3 sentences) that:
1. Acknowledges their loyalty
2. Presents the offer naturally
3. Includes a clear call-to-action

Write a complete, ready-to-send message."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating message: {e}"
    
    def _build_explanation_prompt(
        self,
        decision: Dict[str, Any],
        inputs: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> str:
        """Build a prompt for decision explanation."""
        context_str = ""
        if user_context:
            context_str = f"\n\nAdditional Context:\n{self._format_dict(user_context)}"
        
        prompt = f"""Explain the following business decision in clear, simple language:

Decision Made: {decision['decision']}
Reason: {decision['reason']}
Expected Value: ${decision.get('expected_value', 0):.2f}

Input Metrics:
{self._format_dict(inputs)}{context_str}

Provide a 2-3 sentence explanation that:
1. States what action to take
2. Explains why this maximizes business value
3. Mentions key risk factors

Keep it concise and actionable."""
        return prompt
    
    def _format_dict(self, data: Dict[str, Any]) -> str:
        """Format a dictionary for display in prompts."""
        lines = []
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                lines.append(f"- {formatted_key}: {value:.3f}")
            else:
                lines.append(f"- {formatted_key}: {value}")
        return "\n".join(lines)
    
    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """Parse a structured response into components."""
        parts = {'summary': '', 'rationale': '', 'suggested_actions': ''}
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('SUMMARY:'):
                current_section = 'summary'
                parts['summary'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('RATIONALE:'):
                current_section = 'rationale'
                parts['rationale'] = line.replace('RATIONALE:', '').strip()
            elif line.startswith('SUGGESTED_ACTIONS:'):
                current_section = 'suggested_actions'
                parts['suggested_actions'] = line.replace('SUGGESTED_ACTIONS:', '').strip()
            elif current_section and line:
                parts[current_section] += ' ' + line
        
        return parts


class AIInsightsGenerator:
    """Generates strategic insights from decision patterns using AI."""
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
    
    def analyze_decision_patterns(
        self,
        decision_history: List[Dict[str, Any]],
        time_period: str = "recent activity"
    ) -> Dict[str, Any]:
        """Analyze patterns in decision-making and provide strategic insights."""
        total_decisions = len(decision_history)
        decision_counts = {}
        total_expected_value = 0
        
        for record in decision_history:
            decision = record['decision']['decision']
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            total_expected_value += record['decision'].get('expected_value', 0)
        
        stats_summary = f"""Decision Statistics for {time_period}:
- Total Decisions: {total_decisions}
- Decision Breakdown: {decision_counts}
- Total Expected Value: ${total_expected_value:.2f}
- Average Value per Decision: ${total_expected_value / max(total_decisions, 1):.2f}"""

        prompt = f"""Analyze these decision-making patterns and provide strategic insights:

{stats_summary}

Provide:
1. KEY TRENDS: What patterns do you see?
2. OPPORTUNITIES: Where can we improve ROI?
3. RISKS: What concerns should we monitor?
4. RECOMMENDATIONS: 3 specific actions to optimize decision-making

Be specific and actionable."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'analysis': response['message']['content'],
                'statistics': {
                    'total_decisions': total_decisions,
                    'decision_counts': decision_counts,
                    'total_expected_value': total_expected_value
                }
            }
        except Exception as e:
            return {
                'analysis': f'Error generating insights: {e}',
                'statistics': {}
            }