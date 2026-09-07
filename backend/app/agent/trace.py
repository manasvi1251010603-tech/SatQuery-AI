from typing import Any


def create_trace(
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create the initial observable execution trace
    for a SatQuery analysis plan.
    """

    return [
        {
            "step": "query_interpretation",
            "status": "completed",
            "message": (
                f"Task identified: {plan['task']}"
            ),
        },
        {
            "step": "planning",
            "status": "completed",
            "message": plan["reason"],
        },
        {
            "step": "tool_selection",
            "status": "completed",
            "message": (
                f"Selected tool: {plan['tool']}"
            ),
        },
    ]