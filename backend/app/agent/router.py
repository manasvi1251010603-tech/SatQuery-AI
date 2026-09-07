from backend.app.agent.planner import create_task_plan


def classify_query(
    query: str,
    image_count: int,
) -> dict:
    """
    Public routing interface used by the API.
    """

    return create_task_plan(
        query=query,
        image_count=image_count,
    )