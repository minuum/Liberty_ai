import sqlite3
from typing import Dict, List
import json
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    _instance = None
    _initialized = False
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = "liberty_agent/data/chat.db"):
        if not self._initialized:
            logger.info("======================= DatabaseManager 초기화 시작 =======================")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.db_path = db_path
            self._init_db()
            self._initialized = True
    
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
                
                # 세션이 이미 존재하는지 확인
                cursor.execute("""
                    SELECT session_id FROM chat_sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                if cursor.fetchone() is None:
                    # 새 세션 저장
                    cursor.execute("""
                        INSERT INTO chat_sessions (session_id, user_id, title)
                        VALUES (?, ?, ?)
                    """, (session_id, user_id, title))
                    conn.commit()
                    logger.info(f"새 채팅 세션 저장 완료: {session_id}")
                else:
                    # 기존 세션 제목 업데이트
                    if title:
                        cursor.execute("""
                            UPDATE chat_sessions 
                            SET title = ?
                            WHERE session_id = ?
                        """, (title, session_id))
                        conn.commit()
                        logger.info(f"채팅 세션 제목 업데이트 완료: {session_id}")
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
                    created_at = row['created_at']
                    # 필요 시 datetime 형식으로 변환
                    # created_at = datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
                    sessions.append({
                        'session_id': row['session_id'],
                        'created_at': created_at,
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
                    timestamp = row['timestamp']
                    # 필요 시 datetime 형식으로 변환
                    # timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    message = {
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': timestamp
                    }
                    if row['metadata']:
                        message['metadata'] = json.loads(row['metadata'])
                    messages.append(message)
                return messages
                
        except Exception as e:
            logger.error(f"채팅 기록 조회 중 오류: {str(e)}")
            return []

    def load_chat_history(self, user_id: str, session_id: str) -> List[Dict]:
        """세션의 채팅 기록을 불러옴"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        message_type as role,
                        content
                    FROM chat_messages 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp ASC
                """, (user_id, session_id))
                
                return [{'role': row['role'], 'content': row['content']} 
                        for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"채팅 기록 로드 중 오류: {str(e)}")
            return []
        
    def save_session(self, user_id: str, session_id: str, messages: List[Dict]):
        """세션 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 세션 정보 저장/업데이트
                title = self._generate_session_title(messages)
                cursor.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, title)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_updated = CURRENT_TIMESTAMP,
                        title = ?
                """, (session_id, user_id, title, title))
                
                # 기존 메시지 삭제 (세션 갱신을 위해)
                cursor.execute("""
                    DELETE FROM chat_messages 
                    WHERE session_id = ? AND user_id = ?
                """, (session_id, user_id))
                
                # 새 메시지 저장
                for msg in messages:
                    cursor.execute("""
                        INSERT INTO chat_messages 
                        (user_id, session_id, message_type, content, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        session_id,
                        msg.get("role"),
                        msg.get("content"),
                        json.dumps(msg.get("metadata")) if msg.get("metadata") else None
                    ))
                
                conn.commit()
                logger.info(f"세션 및 메시지 저장 완료 (세션 ID: {session_id})")
                
        except Exception as e:
            logger.error(f"세션 저장 중 오류: {str(e)}")
            raise

    def _generate_session_title(self, messages: List[Dict]) -> str:
        """세션의 제목 생성 (첫 번째 사용자 메시지 사용)"""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content")[:20]  # 첫 번째 사용자 메시지의 앞 20자를 제목으로 사용
        return "새로운 상담 세션"