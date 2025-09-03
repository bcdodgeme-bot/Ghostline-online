# modules/backup_maintenance.py
# Automated Backup & Maintenance for Ghostline AI
# Railway-compatible database backup and knowledge reindexing

import os
import datetime
import json
import gzip
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Any
import subprocess
import psycopg2
from contextlib import contextmanager

from modules.database import get_db_connection, get_database_status
from modules.brain import enhanced_retrieve, get_brain_status

class GhostlineBackupManager:
    """Automated backup and maintenance for Ghostline"""
    
    def __init__(self):
        self.backup_dir = "backups"
        self.database_url = os.getenv('DATABASE_URL')
        self.railway_token = os.getenv('RAILWAY_TOKEN')  # For Railway CLI backups
        self.max_local_backups = 7  # Keep 7 days of backups
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def create_database_backup(self) -> Dict[str, Any]:
        """Create database backup with multiple strategies"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        result = {
            'timestamp': timestamp,
            'success': False,
            'backup_files': [],
            'method': 'unknown',
            'size_bytes': 0,
            'error': None
        }
        
        try:
            # Strategy 1: PostgreSQL dump (preferred for Railway)
            if self.database_url:
                dump_result = self._create_pg_dump(timestamp)
                if dump_result['success']:
                    result.update(dump_result)
                    result['method'] = 'pg_dump'
                    return result
            
            # Strategy 2: Manual table export (fallback)
            manual_result = self._create_manual_backup(timestamp)
            if manual_result['success']:
                result.update(manual_result)
                result['method'] = 'manual_export'
                return result
            
            result['error'] = "All backup strategies failed"
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def _create_pg_dump(self, timestamp: str) -> Dict[str, Any]:
        """Create PostgreSQL dump using pg_dump"""
        backup_file = os.path.join(self.backup_dir, f"ghostline_db_{timestamp}.sql.gz")
        
        try:
            # Parse DATABASE_URL for connection parameters
            from urllib.parse import urlparse
            parsed = urlparse(self.database_url)
            
            # Build pg_dump command
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            cmd = [
                'pg_dump',
                '--host', parsed.hostname,
                '--port', str(parsed.port or 5432),
                '--username', parsed.username,
                '--dbname', parsed.path[1:],  # Remove leading slash
                '--no-password',
                '--verbose',
                '--clean',
                '--if-exists',
                '--create',
                '--format=plain'
            ]
            
            print(f"Running pg_dump command: {' '.join(cmd[:8])}...")  # Don't log full command with credentials
            
            # Run pg_dump and compress
            with open(backup_file, 'wb') as f:
                with gzip.GzipFile(fileobj=f, mode='wb') as gz:
                    process = subprocess.run(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=600  # 10 minute timeout
                    )
                    
                    if process.returncode == 0:
                        gz.write(process.stdout)
                        print(f"pg_dump successful, output size: {len(process.stdout)} bytes")
                    else:
                        print(f"pg_dump failed: {process.stderr.decode()}")
                        return {'success': False, 'error': process.stderr.decode()}
            
            # Verify backup file
            if os.path.exists(backup_file) and os.path.getsize(backup_file) > 0:
                file_size = os.path.getsize(backup_file)
                print(f"Database backup created: {backup_file} ({file_size:,} bytes)")
                
                return {
                    'success': True,
                    'backup_files': [backup_file],
                    'size_bytes': file_size
                }
            else:
                return {'success': False, 'error': 'Backup file was not created or is empty'}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'pg_dump timed out after 10 minutes'}
        except FileNotFoundError:
            print("pg_dump not found, falling back to manual backup")
            return {'success': False, 'error': 'pg_dump command not available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_manual_backup(self, timestamp: str) -> Dict[str, Any]:
        """Manual backup by exporting table data as JSON"""
        backup_file = os.path.join(self.backup_dir, f"ghostline_manual_{timestamp}.json.gz")
        
        try:
            backup_data = {
                'timestamp': timestamp,
                'version': '1.0',
                'tables': {}
            }
            
            with get_db_connection() as conn:
                if not conn:
                    return {'success': False, 'error': 'No database connection'}
                
                cursor = conn.cursor()
                
                # Define tables to backup
                tables_to_backup = [
                    'chat_threads',
                    'uploaded_files', 
                    'daily_logs',
                    'user_settings',
                    'brain_documents',
                    'brain_health',
                    'telegram_reminders'  # If exists
                ]
                
                total_rows = 0
                
                for table in tables_to_backup:
                    try:
                        # Check if table exists
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = %s
                            );
                        """, (table,))
                        
                        if not cursor.fetchone()[0]:
                            print(f"Table {table} does not exist, skipping")
                            continue
                        
                        # Get table data
                        cursor.execute(f"SELECT * FROM {table}")
                        rows = cursor.fetchall()
                        
                        # Get column names
                        cursor.execute("""
                            SELECT column_name FROM information_schema.columns 
                            WHERE table_name = %s ORDER BY ordinal_position
                        """, (table,))
                        
                        columns = [row[0] for row in cursor.fetchall()]
                        
                        # Convert to list of dictionaries
                        table_data = []
                        for row in rows:
                            row_dict = {}
                            for i, value in enumerate(row):
                                # Handle datetime objects
                                if isinstance(value, datetime.datetime):
                                    row_dict[columns[i]] = value.isoformat()
                                elif isinstance(value, datetime.date):
                                    row_dict[columns[i]] = value.isoformat()
                                else:
                                    row_dict[columns[i]] = value
                            table_data.append(row_dict)
                        
                        backup_data['tables'][table] = {
                            'columns': columns,
                            'rows': table_data,
                            'count': len(table_data)
                        }
                        
                        total_rows += len(table_data)
                        print(f"Backed up table {table}: {len(table_data)} rows")
                        
                    except Exception as e:
                        print(f"Failed to backup table {table}: {e}")
                        backup_data['tables'][table] = {'error': str(e)}
            
            # Write compressed JSON backup
            with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            file_size = os.path.getsize(backup_file)
            print(f"Manual backup completed: {total_rows} total rows, {file_size:,} bytes")
            
            return {
                'success': True,
                'backup_files': [backup_file],
                'size_bytes': file_size,
                'total_rows': total_rows
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_brain_backup(self) -> Dict[str, Any]:
        """Create backup of brain/knowledge base"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f"ghostline_brain_{timestamp}.json.gz")
        
        try:
            brain_data = {
                'timestamp': timestamp,
                'version': '1.0',
                'brain_status': get_brain_status(),
                'documents': []
            }
            
            # Export brain documents from database
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT document_id, title, content, chunk_index, metadata
                        FROM brain_documents
                        ORDER BY document_id, chunk_index
                    ''')
                    
                    rows = cursor.fetchall()
                    for row in rows:
                        brain_data['documents'].append({
                            'id': row[0],
                            'title': row[1],
                            'content': row[2],
                            'chunk_index': row[3],
                            'metadata': row[4]
                        })
            
            # Also backup any corpus files
            corpus_files = []
            for ext in ['.jsonl', '.jsonl.gz']:
                pattern = f"data/cleaned/*{ext}"
                import glob
                corpus_files.extend(glob.glob(pattern))
            
            brain_data['corpus_files'] = []
            for corpus_file in corpus_files:
                try:
                    with open(corpus_file, 'rb') as f:
                        brain_data['corpus_files'].append({
                            'filename': os.path.basename(corpus_file),
                            'size': os.path.getsize(corpus_file),
                            'modified': datetime.datetime.fromtimestamp(os.path.getmtime(corpus_file)).isoformat()
                        })
                except Exception as e:
                    print(f"Failed to backup corpus file {corpus_file}: {e}")
            
            # Write compressed brain backup
            with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
                json.dump(brain_data, f, indent=2, default=str)
            
            file_size = os.path.getsize(backup_file)
            document_count = len(brain_data['documents'])
            
            return {
                'success': True,
                'backup_file': backup_file,
                'size_bytes': file_size,
                'document_count': document_count
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_full_system_backup(self) -> Dict[str, Any]:
        """Create complete system backup"""
        print("Starting full system backup...")
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'database_backup': None,
            'brain_backup': None,
            'config_backup': None,
            'success': False,
            'total_size_bytes': 0
        }
        
        # Database backup
        print("Creating database backup...")
        db_result = self.create_database_backup()
        results['database_backup'] = db_result
        if db_result['success']:
            results['total_size_bytes'] += db_result['size_bytes']
        
        # Brain backup
        print("Creating brain backup...")
        brain_result = self.create_brain_backup()
        results['brain_backup'] = brain_result
        if brain_result['success']:
            results['total_size_bytes'] += brain_result['size_bytes']
        
        # Config backup (environment variables, etc.)
        print("Creating config backup...")
        config_result = self._backup_configuration()
        results['config_backup'] = config_result
        if config_result['success']:
            results['total_size_bytes'] += config_result['size_bytes']
        
        # Overall success if at least database backup worked
        results['success'] = db_result['success']
        
        if results['success']:
            print(f"Full system backup completed: {results['total_size_bytes']:,} bytes total")
        else:
            print("Full system backup failed")
        
        return results
    
    def _backup_configuration(self) -> Dict[str, Any]:
        """Backup system configuration"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f"ghostline_config_{timestamp}.json")
        
        try:
            config_data = {
                'timestamp': timestamp,
                'environment_variables': {},
                'file_structure': {},
                'version_info': {}
            }
            
            # Backup non-sensitive environment variables
            safe_env_vars = [
                'CHAT_MODEL', 'OPENROUTER_MODEL', 'GOOGLE_CREDENTIALS_PATH', 
                'GOOGLE_TOKEN_PATH', 'WEBHOOK_URL', 'RAILWAY_STATIC_URL',
                'RAILWAY_ENVIRONMENT', 'PORT'
            ]
            
            for var in safe_env_vars:
                if os.getenv(var):
                    config_data['environment_variables'][var] = os.getenv(var)
            
            # Check what integrations are configured (without exposing keys)
            config_data['integrations'] = {
                'telegram': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
                'clickup': bool(os.getenv('CLICKUP_API_TOKEN')),
                'cloze': bool(os.getenv('CLOZE_API_KEY')),
                'elevenlabs': bool(os.getenv('ELEVENLABS_API_KEY')),
                'replicate': bool(os.getenv('REPLICATE_API_TOKEN')),
                'google': os.path.exists('credentials.json') and os.path.exists('token.json')
            }
            
            # File structure info
            important_dirs = ['sessions', 'data', 'backups', 'daily_logs', 'uploads']
            for dir_name in important_dirs:
                if os.path.exists(dir_name):
                    config_data['file_structure'][dir_name] = {
                        'exists': True,
                        'file_count': len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]),
                        'total_size': sum(os.path.getsize(os.path.join(dir_name, f)) 
                                        for f in os.listdir(dir_name) 
                                        if os.path.isfile(os.path.join(dir_name, f)))
                    }
                else:
                    config_data['file_structure'][dir_name] = {'exists': False}
            
            # Write config backup
            with open(backup_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            file_size = os.path.getsize(backup_file)
            
            return {
                'success': True,
                'backup_file': backup_file,
                'size_bytes': file_size
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def cleanup_old_backups(self):
        """Remove old backup files"""
        try:
            if not os.path.exists(self.backup_dir):
                return
            
            # Get all backup files with their creation times
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith('ghostline_'):
                    filepath = os.path.join(self.backup_dir, filename)
                    if os.path.isfile(filepath):
                        backup_files.append((filepath, os.path.getctime(filepath)))
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the most recent backups
            files_to_delete = backup_files[self.max_local_backups:]
            
            for filepath, _ in files_to_delete:
                try:
                    os.remove(filepath)
                    print(f"Deleted old backup: {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"Failed to delete backup {filepath}: {e}")
            
            if files_to_delete:
                print(f"Cleaned up {len(files_to_delete)} old backup files")
            
        except Exception as e:
            print(f"Backup cleanup failed: {e}")
    
    def reindex_knowledge_base(self) -> Dict[str, Any]:
        """Reindex and optimize the knowledge base"""
        print("Starting knowledge base reindexing...")
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'success': False,
            'operations': [],
            'error': None
        }
        
        try:
            with get_db_connection() as conn:
                if not conn:
                    results['error'] = 'No database connection'
                    return results
                
                cursor = conn.cursor()
                
                # 1. Update database statistics
                print("Updating database statistics...")
                cursor.execute('ANALYZE brain_documents')
                results['operations'].append('Database statistics updated')
                
                # 2. Rebuild full-text search indexes
                print("Rebuilding full-text search indexes...")
                cursor.execute('REINDEX INDEX idx_brain_docs_content_fts')
                results['operations'].append('Full-text search index rebuilt')
                
                # 3. Clean up any orphaned or duplicate documents
                print("Cleaning up duplicate documents...")
                cursor.execute('''
                    DELETE FROM brain_documents a USING brain_documents b 
                    WHERE a.id > b.id 
                    AND a.document_id = b.document_id 
                    AND a.chunk_index = b.chunk_index
                ''')
                deleted_dupes = cursor.rowcount
                if deleted_dupes > 0:
                    results['operations'].append(f'Removed {deleted_dupes} duplicate documents')
                
                # 4. Update brain health
                cursor.execute('SELECT COUNT(*) FROM brain_documents')
                doc_count = cursor.fetchone()[0]
                
                cursor.execute('''
                    INSERT INTO brain_health 
                    (last_refresh, total_documents, health_status, error_log)
                    VALUES (CURRENT_TIMESTAMP, %s, 'healthy', 'Knowledge base reindexed')
                ''', (doc_count,))
                
                results['operations'].append(f'Brain health updated (total docs: {doc_count})')
                
                conn.commit()
                results['success'] = True
                
                print(f"Knowledge base reindexing completed: {len(results['operations'])} operations")
                
        except Exception as e:
            results['error'] = str(e)
            print(f"Knowledge base reindexing failed: {e}")
        
        return results
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform full system maintenance"""
        print("Starting automated maintenance...")
        
        maintenance_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'backup_result': None,
            'reindex_result': None,
            'cleanup_completed': False,
            'success': False
        }
        
        try:
            # 1. Create system backup
            maintenance_results['backup_result'] = self.create_full_system_backup()
            
            # 2. Reindex knowledge base
            maintenance_results['reindex_result'] = self.reindex_knowledge_base()
            
            # 3. Clean up old backups
            self.cleanup_old_backups()
            maintenance_results['cleanup_completed'] = True
            
            # Overall success if backup worked
            maintenance_results['success'] = maintenance_results['backup_result']['success']
            
            if maintenance_results['success']:
                print("Automated maintenance completed successfully")
            else:
                print("Automated maintenance completed with errors")
                
        except Exception as e:
            maintenance_results['error'] = str(e)
            print(f"Automated maintenance failed: {e}")
        
        return maintenance_results


class BackupScheduler:
    """Automated backup scheduler"""
    
    def __init__(self, backup_manager: GhostlineBackupManager):
        self.backup_manager = backup_manager
        self.running = False
        self.thread = None
        
        # Schedule configuration
        self.daily_backup_hour = 2  # 2 AM
        self.maintenance_interval_hours = 24  # Every 24 hours
        
    def start_scheduler(self):
        """Start the automated backup scheduler"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.thread.start()
        print("Backup scheduler started")
    
    def stop_scheduler(self):
        """Stop the backup scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Backup scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        last_backup_date = None
        last_maintenance = datetime.datetime.now()
        
        while self.running:
            try:
                now = datetime.datetime.now()
                current_date = now.date()
                
                # Daily backup check (at specified hour)
                if (last_backup_date != current_date and 
                    now.hour >= self.daily_backup_hour):
                    
                    print("Scheduled daily backup starting...")
                    backup_result = self.backup_manager.create_full_system_backup()
                    
                    if backup_result['success']:
                        print(f"Scheduled backup completed: {backup_result['total_size_bytes']:,} bytes")
                        last_backup_date = current_date
                    else:
                        print("Scheduled backup failed")
                
                # Maintenance check (every N hours)
                hours_since_maintenance = (now - last_maintenance).total_seconds() / 3600
                if hours_since_maintenance >= self.maintenance_interval_hours:
                    
                    print("Scheduled maintenance starting...")
                    maintenance_result = self.backup_manager.perform_maintenance()
                    
                    if maintenance_result['success']:
                        print("Scheduled maintenance completed")
                        last_maintenance = now
                    else:
                        print("Scheduled maintenance failed")
                
                # Sleep for 1 hour before next check
                time.sleep(3600)
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(3600)  # Continue after errors


# Global backup manager instance
backup_manager = GhostlineBackupManager()

# Initialize scheduler but don't start automatically (will be started in app.py)
backup_scheduler = BackupScheduler(backup_manager)

def get_backup_status() -> Dict[str, Any]:
    """Get current backup and maintenance status"""
    status = {
        'scheduler_running': backup_scheduler.running,
        'backup_directory': backup_manager.backup_dir,
        'recent_backups': [],
        'database_status': get_database_status(),
        'brain_status': get_brain_status()
    }
    
    # Get list of recent backups
    try:
        if os.path.exists(backup_manager.backup_dir):
            backup_files = []
            for filename in os.listdir(backup_manager.backup_dir):
                if filename.startswith('ghostline_'):
                    filepath = os.path.join(backup_manager.backup_dir, filename)
                    if os.path.isfile(filepath):
                        backup_files.append({
                            'filename': filename,
                            'size_bytes': os.path.getsize(filepath),
                            'created': datetime.datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                        })
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x['created'], reverse=True)
            status['recent_backups'] = backup_files[:10]  # Last 10 backups
            
    except Exception as e:
        status['backup_list_error'] = str(e)
    
    return status

def start_automated_backups():
    """Start the automated backup system"""
    if not backup_scheduler.running:
        backup_scheduler.start_scheduler()
        return True
    return False

def stop_automated_backups():
    """Stop the automated backup system"""
    if backup_scheduler.running:
        backup_scheduler.stop_scheduler()
        return True
    return False