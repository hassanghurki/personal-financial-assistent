"""
db.py — SQLite helpers for the Personal Finance Assistant.

Schema supports multiple users (user_id) from day one so that
retrofitting multi-tenancy later is never needed.
"""

import sqlite3
import logging
import os

logger = logging.getLogger("finance-assistant.db")

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "finance.db")


def init_db(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create (or open) the SQLite database and ensure the schema exists.
    Returns an open connection.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # WAL mode allows concurrent readers + one writer without "database is locked"
    conn.execute("PRAGMA journal_mode=WAL")
    # Wait up to 5 seconds for a lock before raising an error
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT  NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL DEFAULT 'default',
            date        TEXT    NOT NULL,
            description TEXT    NOT NULL,
            amount      REAL    NOT NULL,
            category    TEXT,
            source      TEXT    DEFAULT 'upload',
            ingested_at TEXT    DEFAULT (datetime('now')),
            is_outlier  INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_user_date
            ON transactions(user_id, date);

        -- Remove restrictive unique index if present so manual multi-buys & forced re-adds work
        DROP INDEX IF EXISTS idx_uniq_tx;

        CREATE TABLE IF NOT EXISTS category_cache (
            description TEXT    PRIMARY KEY,
            category    TEXT    NOT NULL
        );
    """)

    try:
        conn.execute("ALTER TABLE transactions ADD COLUMN is_outlier INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    logger.info("DB initialised at %s", path)
    return conn


# ── Auth helpers ───────────────────────────────────────────────────────────────
import hashlib, secrets


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256. Returns (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex(), salt


def create_user(conn: sqlite3.Connection, username: str, password: str) -> dict:
    """Register a new user. Returns {'ok': True} or {'ok': False, 'error': '...'}."""
    username = username.strip()
    if len(username) < 3:
        return {"ok": False, "error": "Username must be at least 3 characters."}
    if len(password) < 6:
        return {"ok": False, "error": "Password must be at least 6 characters."}

    pw_hash, salt = _hash_password(password)
    stored = f"{salt}:{pw_hash}"
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.lower(), stored),
        )
        conn.commit()
        logger.info("create_user: registered '%s'", username)
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Username already exists. Please choose another."}


def verify_user(conn: sqlite3.Connection, username: str, password: str) -> dict:
    """Verify credentials. Returns {'ok': True, 'user_id': str} or {'ok': False, 'error': '...'}."""
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (username.strip().lower(),),
    ).fetchone()

    if not row:
        return {"ok": False, "error": "Invalid username or password."}

    stored = row["password_hash"]
    salt, pw_hash = stored.split(":", 1)
    candidate_hash, _ = _hash_password(password, salt)

    if candidate_hash != pw_hash:
        return {"ok": False, "error": "Invalid username or password."}

    logger.info("verify_user: '%s' authenticated (id=%s)", username, row["id"])
    return {"ok": True, "user_id": str(row["id"]), "username": username.strip().lower()}


def get_connection(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a connection to the existing DB (init_db must have been called first)."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
