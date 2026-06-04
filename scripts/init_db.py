"""Initialise the PostgreSQL database for LangGraph checkpointing."""
from __future__ import annotations

import asyncio
import os
import sys


async def main() -> None:
    """Create the agents database and run AsyncPostgresSaver.setup()."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        print("ERROR: POSTGRES_URL not set", file=sys.stderr)
        sys.exit(1)

    conn_str = postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)

    import asyncpg
    base_url = postgres_url.rsplit("/", 1)[0] + "/postgres"
    db_name = postgres_url.rsplit("/", 1)[-1].split("?")[0]

    try:
        sys_conn = await asyncpg.connect(base_url)
        try:
            exists = await sys_conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname=$1", db_name
            )
            if not exists:
                await sys_conn.execute(f'CREATE DATABASE "{db_name}"')
                print(f"Created database: {db_name}")
            else:
                print(f"Database already exists: {db_name}")
        finally:
            await sys_conn.close()
    except Exception as e:
        print(f"WARNING: could not ensure DB exists: {e}")

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    checkpointer = await AsyncPostgresSaver.afrom_conn_string(conn_str)
    await checkpointer.setup()
    print("AsyncPostgresSaver.setup() complete")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
