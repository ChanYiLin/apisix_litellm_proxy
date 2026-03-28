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

# DEPRECATED: bedrock_instances is superseded by llm_instances.
# _CREATE_TABLE_BEDROCK = """
# CREATE TABLE IF NOT EXISTS bedrock_instances (
#     instance_name    VARCHAR(255) PRIMARY KEY,
#     aws_region_name  VARCHAR(100) NOT NULL,
#     bedrock_base_url TEXT NOT NULL,
#     api_key          TEXT NOT NULL,
#     model_id         VARCHAR(255) NOT NULL DEFAULT 'anthropic.claude-sonnet-4-5-20250929-v1:0',
#     is_active        TINYINT(1) NOT NULL DEFAULT 1,
#     created_at       DATETIME NOT NULL DEFAULT NOW(),
#     updated_at       DATETIME NOT NULL DEFAULT NOW()
# )
# """
#
# _SEED_ROWS_BEDROCK = [
#     {
#         "instance_name": "bedrock-us-east-1",
#         "aws_region_name": "us-east-1",
#         "bedrock_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
#         "api_key": "REPLACE_ME_US_EAST_1",
#         "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
#     },
#     {
#         "instance_name": "bedrock-ap-northeast-1",
#         "aws_region_name": "ap-northeast-1",
#         "bedrock_base_url": "https://bedrock-runtime.ap-northeast-1.amazonaws.com",
#         "api_key": "REPLACE_ME_AP_NORTHEAST_1",
#         "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
#     },
# ]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_instances (
    instance_name    VARCHAR(255) PRIMARY KEY,
    provider         ENUM('bedrock', 'gemini', 'vertex_ai') NOT NULL,
    model_id         VARCHAR(255) NOT NULL,
    display_model    VARCHAR(255) NOT NULL,
    is_active        TINYINT(1) NOT NULL DEFAULT 1,

    -- Bedrock fields (required when provider='bedrock')
    aws_region_name  VARCHAR(100),
    bedrock_base_url TEXT,
    bedrock_api_key  TEXT,

    -- Gemini AI Studio fields (required when provider='gemini')
    gemini_api_key   TEXT,
    gemini_api_base  TEXT,

    -- Vertex AI fields (required when provider='vertex_ai')
    vertex_project     VARCHAR(255),
    vertex_location    VARCHAR(100),
    vertex_credentials TEXT,
    vertex_api_base    TEXT,

    created_at       DATETIME NOT NULL DEFAULT NOW(),
    updated_at       DATETIME NOT NULL DEFAULT NOW()
)
"""

# Seed data. Replace REPLACE_ME_* placeholders with real credentials before running.
_SEED_ROWS = [
    # --- Bedrock (AWS) ---
    {
        "instance_name":   "bedrock-us-east-1",
        "provider":        "bedrock",
        "model_id":        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "display_model":   "claude-sonnet-4-5",
        "aws_region_name": "us-east-1",
        "bedrock_base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "bedrock_api_key": "REPLACE_ME_BEDROCK_US_EAST_1",
    },
    {
        "instance_name":   "bedrock-ap-northeast-1",
        "provider":        "bedrock",
        "model_id":        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "display_model":   "claude-sonnet-4-5",
        "aws_region_name": "ap-northeast-1",
        "bedrock_base_url": "https://bedrock-runtime.ap-northeast-1.amazonaws.com",
        "bedrock_api_key": "REPLACE_ME_BEDROCK_AP_NORTHEAST_1",
    },
    # --- Gemini AI Studio ---
    {
        "instance_name": "gemini-flash-global",
        "provider":      "gemini",
        "model_id":      "gemini-2.0-flash",
        "display_model": "gemini-2.0-flash",
        "gemini_api_key": "REPLACE_ME_GEMINI_API_KEY",
    },
    {
        "instance_name": "gemini-pro-global",
        "provider":      "gemini",
        "model_id":      "gemini-2.5-pro",
        "display_model": "gemini-2.5-pro",
        "gemini_api_key": "REPLACE_ME_GEMINI_API_KEY",
    },
    # --- Vertex AI (multi-region, used with ai-proxy-multi failover) ---
    {
        "instance_name":      "vertex-flash-us",
        "provider":           "vertex_ai",
        "model_id":           "gemini-2.0-flash",
        "display_model":      "vertex/gemini-2.0-flash",
        "vertex_project":     "REPLACE_ME_GCP_PROJECT",
        "vertex_location":    "us-central1",
        "vertex_credentials": "REPLACE_ME_SA_JSON_STRING",
    },
    {
        "instance_name":      "vertex-flash-asia",
        "provider":           "vertex_ai",
        "model_id":           "gemini-2.0-flash",
        "display_model":      "vertex/gemini-2.0-flash",
        "vertex_project":     "REPLACE_ME_GCP_PROJECT",
        "vertex_location":    "asia-northeast1",
        "vertex_credentials": "REPLACE_ME_SA_JSON_STRING",
    },
]

_INSERT_SQL = """
INSERT IGNORE INTO llm_instances
    (instance_name, provider, model_id, display_model,
     aws_region_name, bedrock_base_url, bedrock_api_key,
     gemini_api_key, gemini_api_base,
     vertex_project, vertex_location, vertex_credentials, vertex_api_base)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


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
            print(f"Table 'llm_instances' created (or already exists) in {DB_NAME}")

            for row in _SEED_ROWS:
                await cur.execute(
                    _INSERT_SQL,
                    (
                        row["instance_name"],
                        row["provider"],
                        row["model_id"],
                        row["display_model"],
                        row.get("aws_region_name"),
                        row.get("bedrock_base_url"),
                        row.get("bedrock_api_key"),
                        row.get("gemini_api_key"),
                        row.get("gemini_api_base"),
                        row.get("vertex_project"),
                        row.get("vertex_location"),
                        row.get("vertex_credentials"),
                        row.get("vertex_api_base"),
                    ),
                )
                print(f"  Inserted (or skipped existing): {row['instance_name']}")
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
