"""
db.py – Database initialization and helper functions.
Uses SQLite with a single persistent connection (check_same_thread=False).
Two tables:
  - users    : registered accounts with roles (student / faculty)
  - resources: faculty-uploaded files with metadata
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def get_conn():
    """Return a persistent SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row                                    
    return conn

def init_db():
    """Create tables if they don't already exist."""
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name    TEXT    NOT NULL,
            username     TEXT    UNIQUE NOT NULL,
            password     TEXT    NOT NULL,
            email        TEXT    NOT NULL,
            gender       TEXT,
            age          INTEGER,
            role         TEXT    NOT NULL DEFAULT 'student'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT NOT NULL,
            original_name TEXT NOT NULL,
            description   TEXT,
            uploader      TEXT NOT NULL,
            upload_date   TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def create_user(full_name, username, password_hash, email, gender, age, role):
    """Insert a new user. Returns True on success, False if username exists."""
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute(
            """INSERT INTO users (full_name, username, password, email, gender, age, role)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (full_name, username, password_hash, email, gender, age, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False                                  
    finally:
        conn.close()

def get_user(username):
    """Fetch a user row by username. Returns sqlite3.Row or None."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row

def username_exists(username):
    """Check if username already exists."""
    return get_user(username) is not None

def add_resource(filename, original_name, description, uploader):
    """Record a newly uploaded file in the DB."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO resources (filename, original_name, description, uploader, upload_date)
           VALUES (?, ?, ?, ?, ?)""",
        (filename, original_name, description, uploader, datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    conn.close()

def get_all_resources():
    """Return list of all uploaded resources (newest first)."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM resources ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def delete_resource(resource_id):
    """Delete a resource DB record and return the stored filename for disk deletion."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT filename FROM resources WHERE id = ?", (resource_id,))
    row = c.fetchone()
    if row:
        c.execute("DELETE FROM resources WHERE id = ?", (resource_id,))
        conn.commit()
        conn.close()
        return row["filename"]
    conn.close()
    return None
