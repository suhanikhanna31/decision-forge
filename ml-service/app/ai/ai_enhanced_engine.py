"""
AI-Enhanced Decision Engine (Ollama version - FREE)
Wraps the core DecisionEngine with AI capabilities.
"""

from typing import Dict, Any, List, Optional
from app.core.decision_engine import DecisionEngine
from app.ai.ai_explainer import AIExplainer, AIInsightsGenerator
import logging

logger = logging.getLogger(__name__)


class AIEnhancedDecisionEngine:
    """
    Enhanced decision engine with AI-powered explanations using Ollama.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        enable_ai: bool = True,
        model: str = "llama3.2"
    ):
        """Initialize the AI-enhanced decision engine."""
        self.base_engine = DecisionEngine(config)
        self.enable_ai = enable_ai
        
        if enable_ai:
            try:
                self.explainer = AIExplainer(model=model)
                self.insights_generator = AIInsightsGenerator(model=model)
                logger.info(f"AI features enabled with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize AI: {e}. Falling back to base engine.")
                self.enable_ai = False
        
        self.decision_history = []
    
    def decide_with_explanation(
        self,
        inputs: Dict[str, Any],
        return_explanation: bool = True,
        return_recommendations: bool = False,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make a decision and optionally return AI-generated explanation."""
        # Get base decision
        decision = self.base_engine.decide(inputs)
        
        result = {
            'decision': decision,
            'inputs': inputs
        }
        
        # Add to history
        self.decision_history.append(result.copy())
        
        # Generate AI enhancements if enabled
        if self.enable_ai:
            try:
                if return_explanation:
                    explanation = self.explainer.explain_decision(
                        decision, inputs, user_context
                    )
                    result['explanation'] = explanation
                
                if return_recommendations and decision['decision'] in ['INTERVENE', 'FLAG']:
                    recommendations = self.explainer.explain_intervention_recommendation(
                        decision,
                        inputs,
                        decision.get('expected_value', 0)
                    )
                    result['recommendations'] = recommendations
            
            except Exception as e:
                logger.error(f"AI enhancement failed: {e}")
                result['ai_error'] = str(e)
        
        return result
    
    def generate_intervention_package(
        self,
        inputs: Dict[str, Any],
        customer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a complete intervention package."""
        result = self.decide_with_explanation(
            inputs,
            return_explanation=True,
            return_recommendations=True
        )
        
        decision = result['decision']
        
        # Generate personalized message if intervention recommended
        if self.enable_ai and decision['decision'] == 'INTERVENE':
            try:
                offer_details = self._determine_offer(
                    decision.get('expected_value', 0),
                    inputs
                )
                
                personalized_message = self.explainer.generate_personalized_message(
                    customer_profile,
                    intervention_type='retention',
                    offer_details=offer_details
                )
                
                result['personalized_message'] = personalized_message
                result['offer_details'] = offer_details
            
            except Exception as e:
                logger.error(f"Message generation failed: {e}")
                result['message_error'] = str(e)
        
        return result
    
    def batch_process_with_explanations(
        self,
        user_inputs: List[Dict[str, Any]],
        include_explanations: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple users and generate decisions with explanations."""
        results = []
        
        for inputs in user_inputs:
            result = self.decide_with_explanation(
                inputs,
                return_explanation=include_explanations
            )
            results.append(result)
        
        return results
    
    def get_strategic_insights(
        self,
        time_period: str = "recent activity",
        min_decisions: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Generate strategic insights from decision history."""
        if not self.enable_ai:
            logger.warning("AI features not enabled")
            return None
        
        if len(self.decision_history) < min_decisions:
            logger.warning(f"Insufficient history: {len(self.decision_history)} < {min_decisions}")
            return None
        
        try:
            insights = self.insights_generator.analyze_decision_patterns(
                self.decision_history,
                time_period=time_period
            )
            return insights
        
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return None
    
    def _determine_offer(
        self,
        expected_value: float,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine appropriate offer."""
        churn_prob = inputs.get('churn', 0)
        
        if expected_value > 1000 or churn_prob > 0.8:
            return {"discount": "25%", "duration": "6 months"}
        elif expected_value > 500 or churn_prob > 0.6:
            return {"discount": "20%", "duration": "3 months"}
        else:
            return {"discount": "15%", "duration": "2 months"}
    
    def get_decision_history_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.decision_history:
            return {"message": "No decision history available"}
        
        decision_counts = {}
        total_value = 0
        
        for record in self.decision_history:
            decision = record['decision']['decision']
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            total_value += record['decision'].get('expected_value', 0)
        
        return {
            "total_decisions": len(self.decision_history),
            "decision_breakdown": decision_counts,
            "total_expected_value": total_value,
            "average_value": total_value / len(self.decision_history)
        }
    
    def clear_history(self):
        """Clear decision history."""
        self.decision_history = []