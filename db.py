import os
from typing import Optional

import aiomysql
from fastapi import HTTPException

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "litellm")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "litellm")

_cache: dict[str, dict] = {}

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


async def _connect() -> aiomysql.Connection:
    return await aiomysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        cursorclass=aiomysql.DictCursor,
        autocommit=True,
    )


async def init_db() -> None:
    conn = await _connect()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_CREATE_TABLE)
    finally:
        conn.close()
    await reload_cache()


async def reload_cache() -> None:
    conn = await _connect()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM llm_instances WHERE is_active = 1")
            rows = await cur.fetchall()
    finally:
        conn.close()
    _cache.clear()
    for row in rows:
        _cache[row["instance_name"]] = row


async def get_instance_config(instance_name: str) -> dict:
    if instance_name in _cache:
        return _cache[instance_name]
    # fallback: direct DB query (cache may be stale after a reload)
    conn = await _connect()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM llm_instances WHERE instance_name = %s AND is_active = 1",
                (instance_name,),
            )
            row = await cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Instance '{instance_name}' not found or inactive",
        )
    _cache[instance_name] = row
    return row
