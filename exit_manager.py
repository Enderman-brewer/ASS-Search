
import sqlite3
import json
import logging
from typing import Dict, Any, Tuple, List, Set

logger = logging.getLogger(__name__)

class ExitManager:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_db()

    def _setup_db(self):
        cursor = self._conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exit_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        self._conn.commit()

    def save_state(self,
                     crawl_queue: List[Tuple[float, int, str]],
                     visited: Set[str],
                     enqueued: Set[str],
                     redirect_cache: Dict[str, Tuple[str, int, float]],
                     per_host_count: Dict[str, int]):
        logger.info("Saving application state to exit.db...")
        try:
            with self._conn:
                self._conn.execute("DELETE FROM exit_state")
                self._conn.execute("INSERT INTO exit_state (key, value) VALUES (?, ?)",
                                   ("crawl_queue", json.dumps(crawl_queue)))
                self._conn.execute("INSERT INTO exit_state (key, value) VALUES (?, ?)",
                                   ("visited", json.dumps(list(visited))))
                self._conn.execute("INSERT INTO exit_state (key, value) VALUES (?, ?)",
                                   ("enqueued", json.dumps(list(enqueued))))
                self._conn.execute("INSERT INTO exit_state (key, value) VALUES (?, ?)",
                                   ("redirect_cache", json.dumps(redirect_cache)))
                self._conn.execute("INSERT INTO exit_state (key, value) VALUES (?, ?)",
                                   ("per_host_count", json.dumps(per_host_count)))
            logger.info("Application state saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save application state: {e}")

    def load_state(self) -> Dict[str, Any]:
        logger.info("Loading application state from exit.db...")
        state = {}
        try:
            with self._conn:
                cursor = self._conn.execute("SELECT key, value FROM exit_state")
                for key, value in cursor.fetchall():
                    state[key] = json.loads(value)
            logger.info("Application state loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load application state: {e}")
        return state
