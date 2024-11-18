import sqlite3
from typing import Dict, List
import json
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "liberty_agent/data/chat.db"):
        """데이터베이스 매니저 초기화"""
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 채팅 세션 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT
                    )
                """)
                
                # 채팅 메시지 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_sessions 
                    ON chat_sessions(user_id, created_at DESC)
                """)
                conn.commit()
                logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            raise

    def save_chat_session(self, user_id: str, session_id: str, title: str = None) -> bool:
        """새로운 채팅 세션 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, title)
                    VALUES (?, ?, ?)
                """, (session_id, user_id, title))
                conn.commit()
                logger.info(f"새 채팅 세션 저장 완료: {session_id}")
                return True
        except Exception as e:
            logger.error(f"채팅 세션 저장 중 오류: {str(e)}")
            return False

    def save_message(self, user_id: str, session_id: str, 
                    message_type: str, content: str, metadata: Dict = None):
        """메시지 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 메시지 저장
                cursor.execute("""
                    INSERT INTO chat_messages 
                    (user_id, session_id, message_type, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, session_id, message_type, content, 
                     json.dumps(metadata) if metadata else None))
                
                # 세션 업데이트 시간 갱신
                cursor.execute("""
                    UPDATE chat_sessions 
                    SET last_updated = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                logger.debug(f"메시지 저장 완료: {session_id}")
        except Exception as e:
            logger.error(f"메시지 저장 실패: {str(e)}")
            raise

    def get_chat_sessions(self, user_id: str) -> List[Dict]:
        """사용자의 모든 채팅 세션 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        session_id,
                        created_at,
                        title
                    FROM chat_sessions 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row['session_id'],
                        'created_at': datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S'),
                        'title': row['title']
                    })
                return sessions
                
        except Exception as e:
            logger.error(f"채팅 세션 조회 중 오류: {str(e)}")
            return []

    def update_session_title(self, session_id: str, title: str) -> bool:
        """채팅 세션 제목 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chat_sessions 
                    SET title = ?
                    WHERE session_id = ?
                """, (title, session_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"세션 제목 업데이트 중 오류: {str(e)}")
            return False

    def get_chat_history(self, user_id: str, session_id: str) -> List[Dict]:
        """특정 세션의 채팅 기록 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        message_type as role,
                        content,
                        metadata,
                        timestamp
                    FROM chat_messages 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp ASC
                """, (user_id, session_id))
                
                messages = []
                for row in cursor.fetchall():
                    message = {
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    }
                    if row['metadata']:
                        message['metadata'] = json.loads(row['metadata'])
                    messages.append(message)
                return messages
                
        except Exception as e:
            logger.error(f"채팅 기록 조회 중 오류: {str(e)}")
            return []   