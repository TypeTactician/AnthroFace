"""SQLite session history management for facial analysis sessions."""

import sqlite3
import base64
import json
import os
import numpy as np
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.db")


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            sex TEXT,
            ethnicity TEXT,
            overall_score REAL,
            symmetry_score REAL,
            proportions_score REAL,
            profile_score REAL,
            golden_ratio_score REAL,
            metrics_json TEXT,
            front_image_b64 TEXT,
            profile_image_b64 TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_session(
    session_id: str,
    sex: str,
    ethnicity: str,
    overall_score: float,
    symmetry_score: float,
    proportions_score: float,
    profile_score: float,
    golden_ratio_score: float,
    metrics: list[dict],
    front_image_b64: Optional[str] = None,
    profile_image_b64: Optional[str] = None,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_at = datetime.now().isoformat()
    metrics_json = json.dumps(metrics, default=_json_default)
    cursor.execute(
        """
        INSERT INTO sessions (
            date, session_id, sex, ethnicity,
            overall_score, symmetry_score, proportions_score,
            profile_score, golden_ratio_score,
            metrics_json, front_image_b64, profile_image_b64, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            date_str,
            session_id,
            sex,
            ethnicity,
            overall_score,
            symmetry_score,
            proportions_score,
            profile_score,
            golden_ratio_score,
            metrics_json,
            front_image_b64,
            profile_image_b64,
            created_at,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_session(session_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def get_all_sessions() -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_metrics(session_id: str) -> Optional[list[dict]]:
    session = get_session(session_id)
    if session is None:
        return None
    return json.loads(session["metrics_json"])


def delete_session(session_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def get_score_history(metric_name: str) -> list[tuple[str, float]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date, overall_score FROM sessions ORDER BY created_at ASC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [(r["date"], r["overall_score"]) for r in rows]


init_db()
