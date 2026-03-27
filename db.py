import os
from typing import Optional

import aiosqlite
from fastapi import HTTPException

DB_PATH = os.environ.get("DB_PATH", "instances.db")

_cache: dict[str, dict] = {}

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


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()
    await reload_cache()


async def reload_cache() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM bedrock_instances WHERE is_active = 1"
        ) as cursor:
            rows = await cursor.fetchall()
    _cache.clear()
    for row in rows:
        _cache[row["instance_name"]] = dict(row)


async def get_instance_config(instance_name: str) -> dict:
    if instance_name in _cache:
        return _cache[instance_name]
    # fallback: direct DB query (cache may be stale after a reload)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM bedrock_instances WHERE instance_name = ? AND is_active = 1",
            (instance_name,),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Instance '{instance_name}' not found or inactive",
        )
    result = dict(row)
    _cache[instance_name] = result
    return result
