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

    def submit_feedback(self, response_id, feedback_type, project=None, user_comment=None):
        """Submit user feedback for a response"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return {'success': False, 'error': 'No database connection'}
                
                with conn.cursor() as cursor:
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
            print(f"Error submitting feedback: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_dashboard_data(self):
        """Get feedback analytics for dashboard"""
        try:
            with get_db_connection() as conn:
                if not conn:
                    return {'total_feedback': 0, 'breakdown': {}}
                
                with conn.cursor() as cursor:
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

# Initialize the global feedback system
_feedback_system = FeedbackSystem()

# STANDALONE FUNCTIONS FOR APP.PY IMPORTS
def submit_user_feedback(response_id, feedback_type, project=None, user_comment=None):
    """Standalone function for app.py import compatibility"""
    return _feedback_system.submit_feedback(response_id, feedback_type, project, user_comment)

def get_feedback_dashboard():
    """Standalone function for app.py import compatibility"""
    return _feedback_system.get_dashboard_data()

# Additional utility functions
def get_feedback_summary(days=7):
    """Get feedback summary for the last N days"""
    try:
        with get_db_connection() as conn:
            if not conn:
                return {}
            
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN feedback_type = 'thumbs_down' THEN 1 ELSE 0 END) as negative,
                        SUM(CASE WHEN feedback_type = 'middle_finger' THEN 1 ELSE 0 END) as sass
                    FROM user_feedback 
                    WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
                ''', (days,))
                
                result = cursor.fetchone()
                if result:
                    total, positive, negative, sass = result
                    return {
                        'total': total or 0,
                        'positive': positive or 0,
                        'negative': negative or 0,
                        'sass': sass or 0,
                        'positive_rate': (positive / total * 100) if total > 0 else 0,
                        'sass_rate': (sass / total * 100) if total > 0 else 0
                    }
                else:
                    return {'total': 0, 'positive': 0, 'negative': 0, 'sass': 0}
                    
    except Exception as e:
        print(f"Error getting feedback summary: {e}")
        return {}

def export_feedback_data(format='json'):
    """Export feedback data for analysis"""
    try:
        with get_db_connection() as conn:
            if not conn:
                return None
            
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT response_id, feedback_type, project, user_comment, created_at
                    FROM user_feedback 
                    ORDER BY created_at DESC
                ''')
                
                data = cursor.fetchall()
                
                if format == 'json':
                    import json
                    return json.dumps([{
                        'response_id': row[0],
                        'feedback_type': row[1],
                        'project': row[2],
                        'comment': row[3],
                        'timestamp': row[4].isoformat() if row[4] else None
                    } for row in data], indent=2)
                else:
                    return data
                    
    except Exception as e:
        print(f"Error exporting feedback data: {e}")
        return None
