# modules/settings_persistence.py
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
