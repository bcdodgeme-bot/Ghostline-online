# modules/feedback_system.py - RAW DATABASE VERSION (NO CONTEXT MANAGERS)
import os
import datetime
import psycopg2

# Database configuration - direct connection without context managers
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

class FeedbackSystem:
    def __init__(self):
        # Don't create tables on init - do it lazily
        self.tables_created = False
    
    def _get_raw_connection(self):
        """Get raw database connection without context manager"""
        if not DATABASE_URL:
            print("No DATABASE_URL found - feedback will not persist")
            return None
        
        try:
            conn = psycopg2.connect(DATABASE_URL)
            return conn
        except Exception as e:
            print(f"Database connection failed: {e}")
            return None
    
    def _ensure_tables(self):
        """Create tables only when needed, with raw database connection"""
        if self.tables_created:
            return True
            
        conn = None
        try:
            conn = self._get_raw_connection()
            if not conn:
                print("No database connection - feedback will not persist")
                return False
            
            cursor = conn.cursor()
            
            # Create feedback table
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
            
            # Create analytics table
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
            
            # Create indexes safely
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_feedback_response_id ON user_feedback (response_id)',
                'CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback (feedback_type)',
                'CREATE INDEX IF NOT EXISTS idx_feedback_project ON user_feedback (project)',
                'CREATE INDEX IF NOT EXISTS idx_analytics_date_project ON feedback_analytics (date, project)'
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as idx_e:
                    print(f"Index creation warning: {idx_e}")
            
            conn.commit()
            cursor.close()
            self.tables_created = True
            print("✅ Feedback system tables created/verified")
            return True
            
        except Exception as e:
            print(f"❌ Error creating feedback tables: {e}")
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def submit_feedback(self, response_id, feedback_type, project=None, user_comment=None):
        """Submit user feedback for a response using raw database connection"""
        # Ensure tables exist
        if not self._ensure_tables():
            return {'success': False, 'error': 'Database initialization failed'}
        
        conn = None
        cursor = None
        try:
            conn = self._get_raw_connection()
            if not conn:
                return {'success': False, 'error': 'No database connection'}
            
            cursor = conn.cursor()
            
            # Insert feedback
            cursor.execute('''
                INSERT INTO user_feedback (response_id, feedback_type, project, user_comment)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            ''', (response_id, feedback_type, project, user_comment))
            
            feedback_id = cursor.fetchone()[0]
            
            # Update analytics
            today = datetime.date.today()
            cursor.execute('''
                INSERT INTO feedback_analytics (date, project, thumbs_up_count, thumbs_down_count, middle_finger_count, total_responses)
                VALUES (%s, %s, 
                        CASE WHEN %s = 'thumbs_up' THEN 1 ELSE 0 END,
                        CASE WHEN %s = 'thumbs_down' THEN 1 ELSE 0 END,
                        CASE WHEN %s = 'middle_finger' THEN 1 ELSE 0 END,
                        1)
                ON CONFLICT (date, project) DO UPDATE SET
                    thumbs_up_count = feedback_analytics.thumbs_up_count + 
                        CASE WHEN %s = 'thumbs_up' THEN 1 ELSE 0 END,
                    thumbs_down_count = feedback_analytics.thumbs_down_count + 
                        CASE WHEN %s = 'thumbs_down' THEN 1 ELSE 0 END,
                    middle_finger_count = feedback_analytics.middle_finger_count + 
                        CASE WHEN %s = 'middle_finger' THEN 1 ELSE 0 END,
                    total_responses = feedback_analytics.total_responses + 1
            ''', (today, project, feedback_type, feedback_type, feedback_type,
                  feedback_type, feedback_type, feedback_type))
            
            conn.commit()
            
            # Map feedback types to emojis for user-friendly response
            emoji_map = {
                'thumbs_up': '👍',
                'thumbs_down': '👎',
                'middle_finger': '🖕'
            }
            
            return {
                'success': True,
                'feedback_id': feedback_id,
                'message': f'Feedback recorded: {emoji_map.get(feedback_type, feedback_type)}'
            }
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            print(f"Error submitting feedback: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def get_dashboard_data(self):
        """Get feedback analytics for dashboard using raw database connection"""
        # Ensure tables exist
        if not self._ensure_tables():
            return {'total_feedback': 0, 'error': 'Database initialization failed'}
        
        conn = None
        cursor = None
        try:
            conn = self._get_raw_connection()
            if not conn:
                return {'total_feedback': 0, 'breakdown': {}}
            
            cursor = conn.cursor()
            
            # Get total feedback count
            cursor.execute('SELECT COUNT(*) FROM user_feedback')
            total_feedback = cursor.fetchone()[0]
            
            # Get feedback breakdown by type
            cursor.execute('''
                SELECT feedback_type, COUNT(*) 
                FROM user_feedback 
                GROUP BY feedback_type
            ''')
            breakdown = dict(cursor.fetchall())
            
            # Get recent feedback (last 7 days)
            cursor.execute('''
                SELECT DATE(created_at) as date, 
                       feedback_type, 
                       COUNT(*) as count
                FROM user_feedback 
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at), feedback_type
                ORDER BY date DESC
            ''')
            recent_feedback = cursor.fetchall()
            
            # Get project breakdown
            cursor.execute('''
                SELECT project, feedback_type, COUNT(*) 
                FROM user_feedback 
                WHERE project IS NOT NULL
                GROUP BY project, feedback_type
                ORDER BY project
            ''')
            project_breakdown = cursor.fetchall()
            
            return {
                'total_feedback': total_feedback,
                'breakdown': breakdown,
                'recent_feedback': recent_feedback,
                'project_breakdown': project_breakdown,
                'emoji_stats': {
                    '👍 Good': breakdown.get('thumbs_up', 0),
                    '👎 Bad': breakdown.get('thumbs_down', 0),
                    '🖕 Sass/Snark': breakdown.get('middle_finger', 0)
                }
            }
            
        except Exception as e:
            print(f"Error getting dashboard data: {e}")
            return {'total_feedback': 0, 'error': str(e)}
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

# Initialize the global feedback system safely
try:
    _feedback_system = FeedbackSystem()
    print("✅ Raw feedback system initialized successfully")
except Exception as e:
    print(f"⚠️ Raw feedback system initialization warning: {e}")
    _feedback_system = None

# STANDALONE FUNCTIONS FOR APP.PY IMPORTS
def submit_user_feedback(response_id, feedback_type, project=None, user_comment=None):
    """Standalone function for app.py import compatibility"""
    if _feedback_system:
        return _feedback_system.submit_feedback(response_id, feedback_type, project, user_comment)
    else:
        return {'success': False, 'error': 'Feedback system not initialized'}

def get_feedback_dashboard():
    """Standalone function for app.py import compatibility"""
    if _feedback_system:
        return _feedback_system.get_dashboard_data()
    else:
        return {'total_feedback': 0, 'error': 'Feedback system not initialized'}

def is_feedback_system_ready():
    """Check if feedback system is properly initialized"""
    return _feedback_system is not None and _feedback_system._ensure_tables()

print("📊 Raw feedback system module loaded successfully")
