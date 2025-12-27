from typing import Any, Dict, List


class Evaluator:
    """Evaluates outputs for completeness and suggests improvements.

    Returns structured scores and suggested next actions.
    """

    def __init__(self):
        pass

    def evaluate(self, goal: str, plan: List[Dict[str, Any]], outputs: Dict[str, Any]) -> Dict[str, Any]:
        # Very simple rule-based evaluator for demo; replace with learned metrics later.
        scores = {
            "relevance": 1.0 if outputs else 0.0,
            "completeness": 0.5,
            "confidence": 0.5,
        }

        suggestions = []
        if outputs.get("description"):
            if len(outputs["description"]) < 20:
                suggestions.append("Provide a more detailed visual description.")
                scores["completeness"] = 0.3

        # Basic aggregated result
        return {
            "scores": scores,
            "suggestions": suggestions,
            "summary": f"Relevance: {scores['relevance']}, Completeness: {scores['completeness']}",
        }
