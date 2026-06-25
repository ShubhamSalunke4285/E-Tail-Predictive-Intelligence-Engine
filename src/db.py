"""Shared access to the live e-commerce application database (SQLite)."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

import config

_engine = create_engine(config.APP_DB_URI)


def get_engine():
    return _engine


def write_table(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    df.to_sql(table, _engine, if_exists=if_exists, index=False)


def read_table(table: str) -> pd.DataFrame:
    return pd.read_sql_table(table, _engine)


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), _engine, params=params or {})


def execute(sql: str, params: dict | None = None) -> None:
    with _engine.begin() as conn:
        conn.execute(text(sql), params or {})


def table_exists(table: str) -> bool:
    from sqlalchemy import inspect
    return table in inspect(_engine).get_table_names()
