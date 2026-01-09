import re
import time
import json
import asyncio
import sqlite3
from typing import Optional


class BaseCacheManager:
    def __init__(self, cache_file: str):
        self.db_path = cache_file
        self.async_lock = asyncio.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    obj TEXT,
                    timestamp REAL,
                    valid INTEGER DEFAULT 1
                )
            """
            )
            conn.commit()

    async def add_to_cache(self, query: str, obj: str):
        if obj is None:
            return
        async with self.async_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, obj, timestamp, valid)
                    VALUES (?, ?, ?, ?)
                """,
                    (query, obj, time.time(), 1),
                )
                conn.commit()

    def hit_cache(self, query: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT obj FROM cache WHERE key = ? AND valid = 1
            """,
                (query,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def in_cache(self, query: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1 FROM cache WHERE key = ? AND valid = 1
            """,
                (query,),
            )
            return cursor.fetchone() is not None

    def invalidate_cache(self, query: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE cache SET valid = 0 WHERE key = ?
            """,
                (query,),
            )
            conn.commit()


class PreprocessCacheManager(BaseCacheManager):
    def preprocess(self, text: str) -> str:
        text = text.strip().lower()
        return re.sub(r"[^\w\s]", "", text)

    async def add_to_cache(self, query: str, obj: dict):
        if obj is None:
            return
        processed_query = self.preprocess(query)
        new_obj = {
            "general": obj["general"],
            "organic": obj["organic"],
        }
        json_obj = json.dumps(new_obj, ensure_ascii=False)
        async with self.async_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # cursor.execute(
                #     """
                #     INSERT OR REPLACE INTO cache (key, obj, timestamp, valid)
                #     VALUES (?, ?, ?, ?, ?)
                # """,
                #     (processed_query, json_obj, time.time(), 1),
                # )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, obj, timestamp, valid)
                    VALUES (?, ?, ?, ?)
                """,
                    (processed_query, json_obj, time.time(), 1),
                )
                conn.commit()

    def hit_cache(self, query: str) -> Optional[dict]:
        processed_query = self.preprocess(query)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT obj FROM cache WHERE key = ? AND valid = 1
            """,
                (processed_query,),
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    def in_cache(self, query: str) -> bool:
        processed_query = self.preprocess(query)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 1 FROM cache WHERE key = ? AND valid = 1
            """,
                (processed_query,),
            )
            return cursor.fetchone() is not None
