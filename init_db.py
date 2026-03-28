"""
Run once to initialise the MariaDB schema and insert seed data.
Usage: python init_db.py
"""
import asyncio
import os

import aiomysql

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "litellm")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "litellm")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bedrock_instances (
    instance_name    VARCHAR(255) PRIMARY KEY,
    aws_region_name  VARCHAR(100) NOT NULL,
    bedrock_base_url TEXT NOT NULL,
    api_key          TEXT NOT NULL,
    model_id         VARCHAR(255) NOT NULL DEFAULT 'anthropic.claude-sonnet-4-5-20250929-v1:0',
    is_active        TINYINT(1) NOT NULL DEFAULT 1,
    created_at       DATETIME NOT NULL DEFAULT NOW(),
    updated_at       DATETIME NOT NULL DEFAULT NOW()
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
    conn = await aiomysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        cursorclass=aiomysql.DictCursor,
        autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(_CREATE_TABLE)
            print(f"Table created (or already exists) in {DB_NAME}")

            for row in _SEED_ROWS:
                await cur.execute(
                    """
                    INSERT IGNORE INTO bedrock_instances
                        (instance_name, aws_region_name, bedrock_base_url, api_key, model_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        row["instance_name"],
                        row["aws_region_name"],
                        row["bedrock_base_url"],
                        row["api_key"],
                        row["model_id"],
                    ),
                )
                print(f"  Inserted (or skipped existing): {row['instance_name']}")
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
