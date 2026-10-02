import sqlite3
import numpy as np
import threading
from typing import List, Dict

class SQLiteStore:
    def __init__(self, db_path: str = "hybridrag.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    text TEXT,
                    chunk_index INTEGER
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector BLOB
                )
            ''')
            conn.commit()

    def save_chunks(self, document_id: str, chunks: List[str], embeddings: List[np.ndarray]):
        with self.lock:
            with self._get_conn() as conn:
                # Delete existing for this doc to allow re-ingestion
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM embeddings WHERE chunk_id LIKE ?", (f"{document_id}_chunk_%",))
                
                chunk_data = []
                emb_data = []
                for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
                    chunk_id = f"{document_id}_chunk_{idx}"
                    chunk_data.append((chunk_id, document_id, text, idx))
                    emb_data.append((chunk_id, emb.tobytes()))
                
                conn.executemany("INSERT INTO chunks (id, document_id, text, chunk_index) VALUES (?, ?, ?, ?)", chunk_data)
                conn.executemany("INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)", emb_data)
                conn.commit()

    def load_all_chunks(self) -> Dict[str, Dict]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT id, document_id, text FROM chunks")
            return {row[0]: {"document_id": row[1], "text": row[2]} for row in cursor}
            
    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT chunk_id, vector FROM embeddings")
            return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor}
