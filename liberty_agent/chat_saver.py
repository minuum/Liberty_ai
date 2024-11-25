import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SqliteSaver:
    def __init__(self, db_path: str = "./chat_logs/chat_history.db"):
        """SQLite 저장소 초기화"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """데이터베이스 및 테이블 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 채팅 이력 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    node_name TEXT NOT NULL,
                    input TEXT,
                    output TEXT,
                    metadata TEXT,
                    chat_type TEXT DEFAULT 'legal'
                )
                """)
                
                # 워크플로우 실행 로그 테이블
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    workflow_path TEXT,
                    total_time FLOAT,
                    final_score FLOAT,
                    metadata TEXT
                )
                """)
                conn.commit()
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 중 오류: {str(e)}")
            raise

    def save_node_execution(self, session_id: str, node_name: str, 
                          input_data: Any, output_data: Any, 
                          metadata: Optional[Dict] = None):
        """노드 실행 결과 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO chat_history 
                (session_id, timestamp, node_name, input, output, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    node_name,
                    str(input_data),
                    str(output_data),
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"노드 실행 결과 저장 중 오류: {str(e)}")

    def save_workflow_execution(self, session_id: str, workflow_path: str,
                              total_time: float, final_score: float,
                              metadata: Optional[Dict] = None):
        """워크플로우 실행 결과 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO workflow_logs 
                (session_id, timestamp, workflow_path, total_time, final_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    workflow_path,
                    total_time,
                    final_score,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"워크플로우 실행 결과 저장 중 오류: {str(e)}") 