# table_discovery.py - Find where conversations are actually stored
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

DATABASE_URL = "postgresql://postgres:gTUnQaRAcOQyZbxNOVSoSTjAVqZqfdZo@centerbeam.proxy.rlwy.net:45686/railway"

@contextmanager
def get_db_connection():
    """Connect to your Railway database"""
    conn = None
    try:
        print(f"🔌 Connecting to Railway database...")
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        print(f"✅ Connected successfully!")
        yield conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        yield None
    finally:
        if conn:
            conn.close()

def discover_conversation_tables():
    """Find all tables that might contain conversations"""
    
    print("🔍 DISCOVERING CONVERSATION TABLES")
    print("=" * 40)
    
    with get_db_connection() as conn:
        if not conn:
            print("❌ Cannot connect to database")
            return
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get all tables in the database
            cursor.execute("""
                SELECT table_name, table_type
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            all_tables = cursor.fetchall()
            
            if all_tables:
                print(f"📊 All tables in database:")
                for table in all_tables:
                    print(f"   - {table['table_name']} ({table['table_type']})")
                
                # Look for conversation-related tables
                conversation_tables = []
                for table in all_tables:
                    table_name = table['table_name'].lower()
                    if any(keyword in table_name for keyword in ['chat', 'conversation', 'message', 'thread', 'history']):
                        conversation_tables.append(table['table_name'])
                
                if conversation_tables:
                    print(f"\n🎯 Potential conversation tables:")
                    for table in conversation_tables:
                        print(f"   ✅ {table}")
                        
                        # Check structure of each potential table
                        try:
                            cursor.execute(f"""
                                SELECT column_name, data_type, is_nullable
                                FROM information_schema.columns 
                                WHERE table_name = '{table}'
                                ORDER BY ordinal_position;
                            """)
                            columns = cursor.fetchall()
                            
                            print(f"      Columns:")
                            for col in columns:
                                print(f"        - {col['column_name']} ({col['data_type']})")
                            
                            # Check if it has data
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            print(f"      Records: {count}")
                            
                            # If it has data, show a sample
                            if count > 0:
                                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                                samples = cursor.fetchall()
                                print(f"      Sample data:")
                                for i, sample in enumerate(samples, 1):
                                    # Show first few fields to identify content
                                    sample_preview = {}
                                    for key, value in sample.items():
                                        if isinstance(value, str) and len(value) > 50:
                                            sample_preview[key] = value[:50] + "..."
                                        else:
                                            sample_preview[key] = value
                                    print(f"        {i}. {sample_preview}")
                            
                            print()  # Empty line between tables
                            
                        except Exception as e:
                            print(f"      ❌ Error examining table: {e}")
                
                else:
                    print(f"\n❌ No obvious conversation tables found")
                    print(f"   Tables might be named differently")
                    
                    # Check each table for conversation-like content
                    print(f"\n🔍 Checking all tables for conversation data:")
                    for table in all_tables:
                        table_name = table['table_name']
                        try:
                            # Get column names
                            cursor.execute(f"""
                                SELECT column_name
                                FROM information_schema.columns 
                                WHERE table_name = '{table_name}'
                                AND data_type IN ('text', 'character varying', 'json', 'jsonb')
                                ORDER BY ordinal_position;
                            """)
                            text_columns = [row['column_name'] for row in cursor.fetchall()]
                            
                            if text_columns:
                                # Check for conversation-like content
                                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                                count = cursor.fetchone()[0]
                                
                                if count > 0:
                                    # Sample some text content
                                    sample_query = f"SELECT {', '.join(text_columns[:3])} FROM {table_name} LIMIT 1"
                                    cursor.execute(sample_query)
                                    sample = cursor.fetchone()
                                    
                                    # Look for conversation indicators
                                    sample_text = str(sample).lower()
                                    conversation_indicators = ['what', 'how', 'tell me', 'syntax', 'hello', 'hi', '?']
                                    
                                    if any(indicator in sample_text for indicator in conversation_indicators):
                                        print(f"   🎯 {table_name} might contain conversations!")
                                        print(f"      Sample: {str(sample)[:100]}...")
                        
                        except Exception as e:
                            continue  # Skip tables we can't read
                            
            else:
                print("❌ No tables found in database")
                
        except Exception as e:
            print(f"❌ Database discovery error: {e}")

if __name__ == "__main__":
    discover_conversation_tables()