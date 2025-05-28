import sqlite3
from contextlib import closing

DB_PATH = 'ballots.db'

# Initialize tables if they don't exist
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS valid_badges (
                badge_id TEXT PRIMARY KEY
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS ballots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                s3_key TEXT UNIQUE,
                session_id TEXT,
                status TEXT,
                badge_id TEXT
            )
        ''')
        # votes
        c.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                category_id TEXT,
                choice TEXT,
                FOREIGN KEY(ballot_id) REFERENCES ballots(id)
            )
        ''')
        conn.commit()


def get_connection():
    return sqlite3.connect(DB_PATH)

def is_valid_badge(badge_id):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute('SELECT 1 FROM valid_badges WHERE badge_id = ?', (badge_id,))
        return c.fetchone() is not None

def insert_ballot(s3_key, session_id, badge_id=None, status='pending'):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute(
            'INSERT OR IGNORE INTO ballots (s3_key, session_id, badge_id, status) VALUES (?, ?, ?, ?)',
            (s3_key, session_id, badge_id, status)
        )
        conn.commit()
        return c.lastrowid

def update_ballot_status(ballot_id, status):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute('UPDATE ballots SET status = ? WHERE id = ?', (status, ballot_id))
        conn.commit()

def insert_vote(category, category_id, choice):
    with closing(get_connection()) as conn:
        c = conn.cursor()
        c.execute(
            'INSERT INTO votes (category, category_id, choice) VALUES (?, ?, ?)',
            (category, category_id, choice)
        )
        conn.commit()
