# modules/hybrid_analysis.py
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
