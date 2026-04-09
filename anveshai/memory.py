"""
Memory System — conversations + progress tracking + spaced repetition.

Schema:
    conversations (id, timestamp, user_input, response)
    progress      (id, timestamp, subject, topic, correct, time_sec, question)
    sr_cards      (id, topic, easiness, interval, repetitions, next_review, last_review)
"""

import sqlite3
import os
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "anveshai_memory.db")
HISTORY_LIMIT = 10


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                user_input  TEXT    NOT NULL,
                response    TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                subject     TEXT    NOT NULL,
                topic       TEXT    NOT NULL,
                correct     INTEGER NOT NULL,
                time_sec    REAL    NOT NULL DEFAULT 0,
                question    TEXT    NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sr_cards (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                topic       TEXT    NOT NULL UNIQUE,
                easiness    REAL    NOT NULL DEFAULT 2.5,
                interval    INTEGER NOT NULL DEFAULT 1,
                repetitions INTEGER NOT NULL DEFAULT 0,
                next_review TEXT    NOT NULL,
                last_review TEXT    NOT NULL
            )
        """)
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────────────────────────────────────

def save_interaction(user_input: str, response: str) -> None:
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (timestamp, user_input, response) VALUES (?, ?, ?)",
            (timestamp, user_input, response),
        )
        conn.commit()


def get_recent_history(limit: int = HISTORY_LIMIT) -> List[sqlite3.Row]:
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, timestamp, user_input, response FROM conversations "
            "ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
    return list(reversed(rows))


def format_history() -> str:
    rows = get_recent_history()
    if not rows:
        return "No conversation history yet."
    lines = [f"  Last {len(rows)} interaction(s):\n"]
    for row in rows:
        lines.append(f"  [{row['timestamp']}]")
        lines.append(f"  You      : {row['user_input']}")
        lines.append(f"  AnveshAI : {row['response']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def clear_history() -> None:
    with _get_connection() as conn:
        conn.execute("DELETE FROM conversations")
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def save_progress(subject: str, topic: str, correct: bool,
                  time_sec: float = 0.0, question: str = "") -> None:
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO progress (timestamp, subject, topic, correct, time_sec, question) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, subject, topic, int(correct), time_sec, question),
        )
        conn.commit()


def get_progress_summary() -> dict:
    """Return dict: subject → {attempted, correct, accuracy, topics: {topic: {attempted, correct}}}"""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT subject, topic, correct, time_sec FROM progress"
        ).fetchall()
    summary: dict = {}
    for row in rows:
        subj = row['subject']
        topic = row['topic']
        if subj not in summary:
            summary[subj] = {'attempted': 0, 'correct': 0, 'time_sec': 0.0, 'topics': {}}
        summary[subj]['attempted'] += 1
        summary[subj]['correct']   += row['correct']
        summary[subj]['time_sec']  += row['time_sec']
        if topic not in summary[subj]['topics']:
            summary[subj]['topics'][topic] = {'attempted': 0, 'correct': 0}
        summary[subj]['topics'][topic]['attempted'] += 1
        summary[subj]['topics'][topic]['correct']   += row['correct']
    for subj in summary:
        a = summary[subj]['attempted']
        c = summary[subj]['correct']
        summary[subj]['accuracy'] = c / a * 100 if a > 0 else 0.0
    return summary


def get_weak_topics(threshold: float = 60.0, min_attempts: int = 1) -> List[Tuple[str, str, float]]:
    """Return [(subject, topic, accuracy%)] for topics below threshold."""
    summary = get_progress_summary()
    weak = []
    for subj, data in summary.items():
        for topic, tdata in data['topics'].items():
            if tdata['attempted'] >= min_attempts:
                acc = tdata['correct'] / tdata['attempted'] * 100
                if acc < threshold:
                    weak.append((subj, topic, acc))
    weak.sort(key=lambda x: x[2])
    return weak


# ─────────────────────────────────────────────────────────────────────────────
# Spaced repetition (SM-2 algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def update_sr_card(topic: str, quality: int) -> None:
    """
    Update SM-2 card for a topic.
    quality: 0-5 (0-1 = blackout, 2 = incorrect but close, 3 = correct+hard, 4 = correct, 5 = easy)
    """
    today = _today()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sr_cards WHERE topic = ?", (topic,)
        ).fetchone()

        if row is None:
            # New card
            easiness = 2.5
            repetitions = 0
            interval = 1
        else:
            easiness    = row['easiness']
            repetitions = row['repetitions']
            interval    = row['interval']

        # SM-2 update
        if quality < 3:
            repetitions = 0
            interval    = 1
        else:
            if repetitions == 0:
                interval = 1
            elif repetitions == 1:
                interval = 6
            else:
                interval = round(interval * easiness)
            repetitions += 1
        easiness = max(1.3, easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        next_review = (date.today() + timedelta(days=interval)).isoformat()

        if row is None:
            conn.execute(
                "INSERT INTO sr_cards (topic, easiness, interval, repetitions, next_review, last_review) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (topic, easiness, interval, repetitions, next_review, today),
            )
        else:
            conn.execute(
                "UPDATE sr_cards SET easiness=?, interval=?, repetitions=?, "
                "next_review=?, last_review=? WHERE topic=?",
                (easiness, interval, repetitions, next_review, today, topic),
            )
        conn.commit()


def get_due_topics() -> List[dict]:
    """Return list of topics due for review today."""
    today = _today()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT topic, easiness, interval, repetitions, next_review, last_review "
            "FROM sr_cards WHERE next_review <= ? ORDER BY next_review",
            (today,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_sr_cards() -> List[dict]:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT topic, easiness, interval, repetitions, next_review, last_review "
            "FROM sr_cards ORDER BY next_review"
        ).fetchall()
    return [dict(r) for r in rows]
