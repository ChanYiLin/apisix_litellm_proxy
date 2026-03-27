"""
Run once to initialise the SQLite DB and insert seed data.
Usage: python init_db.py
"""
import asyncio
import os

import aiosqlite

DB_PATH = os.environ.get("DB_PATH", "instances.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bedrock_instances (
    instance_name    TEXT PRIMARY KEY,
    aws_region_name  TEXT NOT NULL,
    bedrock_base_url TEXT NOT NULL,
    api_key          TEXT NOT NULL,
    model_id         TEXT NOT NULL DEFAULT 'anthropic.claude-sonnet-4-5-20250929-v1:0',
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_SEED_ROWS = [
    {
        "instance_name": "bedrock-us-east-1",
        "aws_region_name": "us-east-1",
        "bedrock_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "api_key": "REPLACE_ME_US_EAST_1",
        "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    },
    {
        "instance_name": "bedrock-ap-northeast-1",
        "aws_region_name": "ap-northeast-1",
        "bedrock_base_url": "https://bedrock-runtime.ap-northeast-1.amazonaws.com",
        "api_key": "REPLACE_ME_AP_NORTHEAST_1",
        "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    },
]


async def main() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()
        print(f"Table created (or already exists) in {DB_PATH}")

        for row in _SEED_ROWS:
            await db.execute(
                """
                INSERT OR IGNORE INTO bedrock_instances
                    (instance_name, aws_region_name, bedrock_base_url, api_key, model_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["instance_name"],
                    row["aws_region_name"],
                    row["bedrock_base_url"],
                    row["api_key"],
                    row["model_id"],
                ),
            )
            await db.commit()
            print(f"  Inserted (or skipped existing): {row['instance_name']}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
