#!/usr/bin/env python3
"""
Force Fix Script - Replace existing modules and fix database setup
"""

import os
import sys
from modules.database import get_db_connection

def force_replace_modules():
    """Force replace the existing problematic modules"""
    print("🔧 Force replacing problematic modules...")
    
    # Force replace settings_persistence.py
    print("   📝 Replacing modules/settings_persistence.py...")
    with open("modules/settings_persistence.py", "w") as f:
        f.write("""# modules/settings_persistence.py
import os
import datetime
import psycopg2
from modules.database import get_db_connection

class SettingsPersistence:
    def __init__(self):
        self._create_tables()
    
    def _create_tables(self):
        with get_db_connection() as conn:
            if not conn:
                print("No database connection - settings will not persist")
                return
            
            try:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS user_settings (
                            id SERIAL PRIMARY KEY,
                            setting_key VARCHAR(100) UNIQUE NOT NULL,
                            setting_value JSONB NOT NULL,
                            setting_type VARCHAR(50) DEFAULT 'general',
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS user_preference_profiles (
                            id SERIAL PRIMARY KEY,
                            profile_name VARCHAR(100) UNIQUE NOT NULL,
                            profile_description TEXT,
                            settings JSONB NOT NULL,
                            is_active BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS settings_history (
                            id SERIAL PRIMARY KEY,
                            setting_key VARCHAR(100) NOT NULL,
                            old_value JSONB,
                            new_value JSONB NOT NULL,
                            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            change_reason VARCHAR(255)
                        )
                    ''')
                    
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_settings_key ON user_settings (setting_key)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_settings_type ON user_settings (setting_type)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_profiles_active ON user_preference_profiles (is_active)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_key_time ON settings_history (setting_key, changed_at)')
                    
                    conn.commit()
                    print("✅ Settings persistence tables created/verified")
                    
            except Exception as e:
                print(f"❌ Error creating settings tables: {e}")
                raise
""")
    
    # Force replace feedback_system.py
    print("   📝 Replacing modules/feedback_system.py...")
    with open("modules/feedback_system.py", "w") as f:
        f.write("""# modules/feedback_system.py
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
""")
    
    # Force replace hybrid_analysis.py
    print("   📝 Replacing modules/hybrid_analysis.py...")
    with open("modules/hybrid_analysis.py", "w") as f:
        f.write("""# modules/hybrid_analysis.py
import os
import datetime
import psycopg2
from modules.database import get_db_connection

class HybridAnalysisEngine:
    def __init__(self):
        self._create_tables()
    
    def _create_tables(self):
        with get_db_connection() as conn:
            if not conn:
                print("No database connection - hybrid analysis will not persist")
                return
            
            try:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS content_strategies (
                            id SERIAL PRIMARY KEY,
                            strategy_name VARCHAR(255) NOT NULL,
                            strategy_type VARCHAR(100),
                            content JSONB NOT NULL,
                            project VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            is_active BOOLEAN DEFAULT TRUE
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS content_performance (
                            id SERIAL PRIMARY KEY,
                            strategy_id INTEGER REFERENCES content_strategies(id),
                            metric_name VARCHAR(100) NOT NULL,
                            metric_value DECIMAL(10,2),
                            measurement_date DATE NOT NULL,
                            notes TEXT
                        )
                    ''')
                    
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_name ON content_strategies (strategy_name)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_type ON content_strategies (strategy_type)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_project ON content_strategies (project)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_strategy ON content_performance (strategy_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_date ON content_performance (measurement_date)')
                    
                    conn.commit()
                    print("✅ Hybrid analysis tables created/verified")
                    
            except Exception as e:
                print(f"❌ Error creating hybrid analysis tables: {e}")
                raise
""")
    
    print("✅ All modules force-replaced successfully!")

def manual_table_creation():
    """Manually create all missing tables using direct SQL"""
    print("🔨 Creating tables manually with direct SQL...")
    
    with get_db_connection() as conn:
        if not conn:
            print("❌ No database connection!")
            return False
        
        try:
            with conn.cursor() as cursor:
                # User feedback table
                print("   📝 Creating user_feedback table...")
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
                
                # Feedback analytics table
                print("   📝 Creating feedback_analytics table...")
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
                
                # User preference profiles table
                print("   📝 Creating user_preference_profiles table...")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_preference_profiles (
                        id SERIAL PRIMARY KEY,
                        profile_name VARCHAR(100) UNIQUE NOT NULL,
                        profile_description TEXT,
                        settings JSONB NOT NULL,
                        is_active BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Settings history table
                print("   📝 Creating settings_history table...")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS settings_history (
                        id SERIAL PRIMARY KEY,
                        setting_key VARCHAR(100) NOT NULL,
                        old_value JSONB,
                        new_value JSONB NOT NULL,
                        changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        change_reason VARCHAR(255)
                    )
                ''')
                
                # Content strategies table
                print("   📝 Creating content_strategies table...")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS content_strategies (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(255) NOT NULL,
                        strategy_type VARCHAR(100),
                        content JSONB NOT NULL,
                        project VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                # Content performance table
                print("   📝 Creating content_performance table...")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS content_performance (
                        id SERIAL PRIMARY KEY,
                        strategy_id INTEGER REFERENCES content_strategies(id),
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value DECIMAL(10,2),
                        measurement_date DATE NOT NULL,
                        notes TEXT
                    )
                ''')
                
                # Create all indexes
                print("   📝 Creating indexes...")
                indexes = [
                    'CREATE INDEX IF NOT EXISTS idx_feedback_response_id ON user_feedback (response_id)',
                    'CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback (feedback_type)',
                    'CREATE INDEX IF NOT EXISTS idx_feedback_project ON user_feedback (project)',
                    'CREATE INDEX IF NOT EXISTS idx_analytics_date_project ON feedback_analytics (date, project)',
                    'CREATE INDEX IF NOT EXISTS idx_settings_key ON user_settings (setting_key)',
                    'CREATE INDEX IF NOT EXISTS idx_settings_type ON user_settings (setting_type)',
                    'CREATE INDEX IF NOT EXISTS idx_profiles_active ON user_preference_profiles (is_active)',
                    'CREATE INDEX IF NOT EXISTS idx_history_key_time ON settings_history (setting_key, changed_at)',
                    'CREATE INDEX IF NOT EXISTS idx_strategies_name ON content_strategies (strategy_name)',
                    'CREATE INDEX IF NOT EXISTS idx_strategies_type ON content_strategies (strategy_type)',
                    'CREATE INDEX IF NOT EXISTS idx_strategies_project ON content_strategies (project)',
                    'CREATE INDEX IF NOT EXISTS idx_performance_strategy ON content_performance (strategy_id)',
                    'CREATE INDEX IF NOT EXISTS idx_performance_date ON content_performance (measurement_date)'
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                conn.commit()
                print("✅ All tables and indexes created successfully!")
                return True
                
        except Exception as e:
            print(f"❌ Manual table creation failed: {e}")
            return False

def verify_all_tables():
    """Verify all tables exist"""
    print("🔍 Verifying all tables exist...")
    
    with get_db_connection() as conn:
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                new_tables = [
                    'user_feedback', 'feedback_analytics',
                    'user_settings', 'user_preference_profiles', 'settings_history',
                    'content_strategies', 'content_performance'
                ]
                
                existing_tables = []
                missing_tables = []
                
                for table in new_tables:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        );
                    """, (table,))
                    
                    if cur.fetchone()[0]:
                        existing_tables.append(table)
                        print(f"   ✅ {table}")
                    else:
                        missing_tables.append(table)
                        print(f"   ❌ {table} - MISSING")
                
                if missing_tables:
                    print(f"\n⚠️  Warning: {len(missing_tables)} tables are missing!")
                    return False
                else:
                    print(f"\n🎉 All {len(existing_tables)} tables verified!")
                    return True
                    
        except Exception as e:
            print(f"❌ Table verification failed: {e}")
            return False

if __name__ == "__main__":
    print("🚀 Force Fix Script - Replacing modules and creating tables")
    print("=" * 60)
    
    # Test database connection first
    print("1. Testing database connection...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Connected to PostgreSQL: {version}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please check your DATABASE_URL environment variable")
        sys.exit(1)
    
    # Force replace modules
    print("\n2. Force replacing modules...")
    force_replace_modules()
    
    # Manual table creation
    print("\n3. Manual table creation...")
    success = manual_table_creation()
    
    if not success:
        print("❌ Manual table creation failed!")
        sys.exit(1)
    
    # Verify tables
    print("\n4. Verifying tables...")
    verified = verify_all_tables()
    
    if verified:
        print("\n🎉 SUCCESS! All tables created and verified!")
        print("\n📋 NEXT STEPS:")
        print("1. Restart your Ghostline application")
        print("2. Test the feedback buttons (👍👎🖕)")
        print("3. Try command: 'hybrid analysis' for content strategy")
        print("4. Check that personality settings persist after restart")
    else:
        print("\n❌ FAILED! Some tables are still missing.")
        sys.exit(1)