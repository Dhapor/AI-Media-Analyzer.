from typing import Any, Dict, List
import sys
import os

# Ensure project root is in Python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Keep internal relative imports
from executor import Executor
from evaluator import Evaluator
from planner import Planner  # use relative or absolute depending on planner.py location

class Agent:
    """Orchestrates Plan -> Act -> Observe -> Reflect loop.

    The run method returns a structured trace for explainability.
    """

    def __init__(self, gemini_client):
        self.planner = Planner()
        self.executor = Executor(gemini_client)
        self.evaluator = Evaluator()

    def run(self, goal: str, inputs: Dict[str, Any], max_iters: int = 3) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        observations = dict(inputs)

        for iteration in range(max_iters):
            step_plan = self.planner.plan(goal, observations)
            iteration_record: Dict[str, Any] = {"plan": step_plan, "actions": [], "observations": []}

            for step in step_plan:
                # pass some context for synthesis steps
                if step["action"] == "synthesize":
                    step["params"]["context"] = observations
                result = self.executor.execute(step)
                iteration_record["actions"].append({"step_id": step["id"], "result": result})

                # Merge observations in simple way
                if isinstance(result, dict):
                    observations.update({k: v for k, v in result.items() if k})
                iteration_record["observations"].append(result)

            # Evaluate iteration outputs
            eval_result = self.evaluator.evaluate(goal, step_plan, observations)
            iteration_record["evaluation"] = eval_result

            trace.append(iteration_record)

            # Reflection / stopping: stop early if evaluator is satisfied
            if eval_result["scores"].get("relevance", 0) > 0.9:
                break

        return {"goal": goal, "final_observations": observations, "trace": trace}
