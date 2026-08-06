from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from app.config import settings


def get_checkpointer() -> PostgresSaver:
    """
    Build a PostgresSaver backed by a connection pool.

    Call this once at application startup and pass the returned instance into
    graph.compile(checkpointer=checkpointer).

    PostgresSaver.setup() creates LangGraph's internal checkpoint tables
    (checkpoints, checkpoint_writes, etc.) the first time it runs; it is safe
    to call on every startup.
    """
    pool = ConnectionPool(
        conninfo=settings.database_url,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer
