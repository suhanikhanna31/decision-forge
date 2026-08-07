"""
Natural Language Query Interface (Ollama version)
"""

from typing import Dict, Any
import ollama


class NLQueryInterface:
    """Natural language interface using Ollama."""
    
    def __init__(self, ai_engine, model: str = "llama3.2"):
        self.ai_engine = ai_engine
        self.model = model
    
    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """Process a natural language query."""
        
        # Get context
        history_summary = self.ai_engine.get_decision_history_summary()
        
        prompt = f"""Answer this question about a customer decision system:

Question: {natural_language_query}

System Context:
- This is an ML-driven system for customer interventions
- Recent decisions: {history_summary.get('total_decisions', 0)}
- Decision types: INTERVENE, DO_NOTHING, FLAG

Provide a clear, helpful answer in 2-3 sentences."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "query": natural_language_query,
                "response": response['message']['content'],
                "type": "general",
                "success": True
            }
        
        except Exception as e:
            return {
                "query": natural_language_query,
                "response": f"Error: {e}",
                "type": "error",
                "success": False
            }