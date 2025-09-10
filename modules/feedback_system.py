# modules/feedback_system.py
import os
import datetime
import psycopg2
from modules.database import get_db_connection

class FeedbackSystem:
    def __init__(self):
        self._create_tables()
    
    def _create_tables(self):
        with get_db_connection() as conn:
            if not conn:
                print("No database connection - feedback will not persist")
                return
            
            try:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS user_feedback (
                            id SERIAL PRIMARY KEY,
                            response_id VARCHAR(255) NOT NULL,
                            feedback_type VARCHAR(20) NOT NULL CHECK (feedback_type IN ('thumbs_up', 'thumbs_down', 'middle_finger')),
                            user_comment TEXT,
                            project VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS feedback_analytics (
                            id SERIAL PRIMARY KEY,
                            date DATE NOT NULL,
                            project VARCHAR(100),
                            thumbs_up_count INTEGER DEFAULT 0,
                            thumbs_down_count INTEGER DEFAULT 0,
                            middle_finger_count INTEGER DEFAULT 0,
                            total_responses INTEGER DEFAULT 0,
                            UNIQUE(date, project)
                        )
                    ''')
                    
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_response_id ON user_feedback (response_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback (feedback_type)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_project ON user_feedback (project)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_date_project ON feedback_analytics (date, project)')
                    
                    conn.commit()
                    print("✅ Feedback system tables created/verified")
                    
            except Exception as e:
                print(f"❌ Error creating feedback tables: {e}")
                raise
