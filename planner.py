from typing import Any, Dict, List

class Planner:
    """Simple, extensible planner.

    Given a user goal and current observations, produce a structured plan:
    list of steps with an action, parameters, and rationale.
    """

    def __init__(self):
        pass

    def plan(self, goal: str, observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        steps = []

        # If there's an image, analyze it first
        if observations.get("image") is not None:
            steps.append({
                "id": "describe_image",
                "action": "describe_image",
                "params": {"image": observations["image"]},
                "rationale": "Extract visual features to ground the plan",
            })

        # Add a synthesis step
        steps.append({
            "id": "synthesize_findings",
            "action": "synthesize",
            "params": {"goal": goal},
            "rationale": "Synthesize observations with the goal to produce conclusions",
        })

        # Optionally request follow-up question generation
        steps.append({
            "id": "suggest_improvements",
            "action": "suggest_improvements",
            "params": {"goal": goal},
            "rationale": "Propose corrections or follow-ups to improve the result",
        })

        return steps
