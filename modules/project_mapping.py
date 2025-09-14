# modules/project_mapping.py
"""
Ghostline Project-Folder Mapping System
Architecture for connecting websites, analytics, and social accounts to project folders
"""
import json
import os
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

class ProjectMappingSystem:
    """Central system for managing project-folder mappings and context routing"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.mapping_cache = {}
        self.current_project_context = "Main"  # Default to main folder
        
        # Core project definitions (matches your existing PROJECTS list)
        self.CORE_PROJECTS = [
            'Personal Operating Manual', 'AMCF', 'BCDodgeme', 'Rose and Angel', 
            'Meals N Feelz', 'TV Signals', 'Damn It Carl', 'HalalBot', 
            'Kitchen', 'Health', 'Side Quests'
        ]
        
        # Initialize database tables
        self._init_mapping_tables()
        
        # Load default mappings
        self._load_default_mappings()
    
    @contextmanager
    def get_db_connection(self):
        """Database connection context manager"""
        conn = None
        try:
            if self.database_url:
                conn = psycopg2.connect(self.database_url)
                conn.autocommit = True
                yield conn
            else:
                yield None
        except Exception as e:
            print(f"Database error: {e}")
            yield None
        finally:
            if conn:
                conn.close()
    
    def _init_mapping_tables(self):
        """Initialize database tables for project mappings"""
        with self.get_db_connection() as conn:
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # Project mappings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_mappings (
                    id SERIAL PRIMARY KEY,
                    project_name VARCHAR(100) NOT NULL,
                    mapping_type VARCHAR(50) NOT NULL,  -- 'website', 'social', 'analytics', 'email'
                    resource_identifier VARCHAR(500) NOT NULL,  -- domain, account, etc.
                    resource_data JSONB DEFAULT '{}',
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_name, mapping_type, resource_identifier)
                )
            ''')
            
            # Project context sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    project_name VARCHAR(100) NOT NULL,
                    context_data JSONB DEFAULT '{}',
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id)
                )
            ''')
            
            # Conversation history with project context
            cursor.execute('''
                ALTER TABLE chat_threads 
                ADD COLUMN IF NOT EXISTS context_project VARCHAR(100),
                ADD COLUMN IF NOT EXISTS context_data JSONB DEFAULT '{}'
            ''')
            
            print("✅ Project mapping tables initialized")
    
    def _load_default_mappings(self):
        """Load default project mappings"""
        default_mappings = {
            'Meals N Feelz': {
                'websites': ['mealsnfeelz.org'],
                'social': ['@mealsnfeelz_instagram', '@mealsnfeelz_facebook'],
                'analytics': ['mealsnfeelz.org_ga4', 'mealsnfeelz.org_gsc'],
                'email': ['info@mealsnfeelz.org']
            },
            'AMCF': {
                'websites': ['amcf.org'],
                'social': ['@amcf_instagram', '@amcf_facebook'],
                'analytics': ['amcf.org_ga4', 'amcf.org_gsc'],
                'email': ['contact@amcf.org']
            },
            'Rose and Angel': {
                'websites': ['roseandangel.com'],
                'social': ['@roseandangel_social'],
                'analytics': ['roseandangel.com_ga4'],
                'email': ['hello@roseandangel.com']
            },
            'BCDodgeme': {
                'websites': ['bcdodgeme.com'],
                'social': ['@bcdodgeme_twitter'],
                'analytics': ['bcdodgeme.com_ga4'],
                'email': ['support@bcdodgeme.com']
            },
            'TV Signals': {
                'websites': ['tvsignals.net'],
                'social': ['@tvsignals_youtube'],
                'analytics': ['tvsignals.net_ga4'],
                'email': ['contact@tvsignals.net']
            }
        }
        
        # Store in database
        for project, mappings in default_mappings.items():
            for mapping_type, resources in mappings.items():
                for resource in resources:
                    self.add_project_mapping(project, mapping_type, resource)
    
    def add_project_mapping(self, project: str, mapping_type: str, resource_identifier: str, 
                          resource_data: Dict = None) -> bool:
        """Add a new project mapping"""
        with self.get_db_connection() as conn:
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO project_mappings 
                    (project_name, mapping_type, resource_identifier, resource_data)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (project_name, mapping_type, resource_identifier) 
                    DO UPDATE SET 
                        resource_data = EXCLUDED.resource_data,
                        is_active = true,
                        created_at = CURRENT_TIMESTAMP
                ''', (project, mapping_type, resource_identifier, json.dumps(resource_data or {})))
                
                # Update cache
                self._refresh_mapping_cache()
                return True
                
            except Exception as e:
                print(f"Error adding mapping: {e}")
                return False
    
    def get_project_mappings(self, project: str = None) -> Dict:
        """Get all mappings for a project or all projects"""
        with self.get_db_connection() as conn:
            if not conn:
                return {}
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if project:
                cursor.execute('''
                    SELECT * FROM project_mappings 
                    WHERE project_name = %s AND is_active = true
                    ORDER BY mapping_type, resource_identifier
                ''', (project,))
            else:
                cursor.execute('''
                    SELECT * FROM project_mappings 
                    WHERE is_active = true
                    ORDER BY project_name, mapping_type, resource_identifier
                ''')
            
            results = cursor.fetchall()
            
            # Organize by project and type
            organized = {}
            for row in results:
                proj = row['project_name']
                map_type = row['mapping_type']
                
                if proj not in organized:
                    organized[proj] = {}
                if map_type not in organized[proj]:
                    organized[proj][map_type] = []
                
                organized[proj][map_type].append({
                    'identifier': row['resource_identifier'],
                    'data': row['resource_data'] or {},
                    'id': row['id']
                })
            
            return organized
    
    def identify_project_from_context(self, context_data: Dict) -> str:
        """Identify project from various context clues"""
        # Check for website/domain mentions
        text_content = str(context_data.get('user_input', '')).lower()
        
        # Website-based identification
        domain_mappings = {
            'mealsnfeelz.org': 'Meals N Feelz',
            'amcf.org': 'AMCF',
            'roseandangel.com': 'Rose and Angel',
            'bcdodgeme.com': 'BCDodgeme',
            'tvsignals.net': 'TV Signals'
        }
        
        for domain, project in domain_mappings.items():
            if domain in text_content:
                return project
        
        # Keyword-based identification
        keyword_mappings = {
            'Meals N Feelz': ['meal', 'food', 'recipe', 'nutrition', 'cooking'],
            'AMCF': ['giving circle', 'nonprofit', 'charity', 'fundraising'],
            'Health': ['health', 'medical', 'fitness', 'wellness'],
            'Kitchen': ['kitchen', 'cooking', 'recipe'],
            'Personal Operating Manual': ['workflow', 'productivity', 'system']
        }
        
        for project, keywords in keyword_mappings.items():
            if any(keyword in text_content for keyword in keywords):
                return project
        
        # Default to current context or Main
        return context_data.get('current_project', 'Main')
    
    def set_project_context(self, session_id: str, project: str, context_data: Dict = None):
        """Set current project context for a session"""
        with self.get_db_connection() as conn:
            if not conn:
                self.current_project_context = project
                return
            
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO project_sessions (session_id, project_name, context_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) 
                DO UPDATE SET 
                    project_name = EXCLUDED.project_name,
                    context_data = EXCLUDED.context_data,
                    last_accessed = CURRENT_TIMESTAMP
            ''', (session_id, project, json.dumps(context_data or {})))
            
            self.current_project_context = project
    
    def get_project_context(self, session_id: str) -> Dict:
        """Get current project context for a session"""
        with self.get_db_connection() as conn:
            if not conn:
                return {'project': self.current_project_context, 'data': {}}
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute('''
                SELECT project_name, context_data 
                FROM project_sessions 
                WHERE session_id = %s
            ''', (session_id,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'project': result['project_name'],
                    'data': result['context_data'] or {}
                }
            
            return {'project': 'Main', 'data': {}}
    
    def filter_conversation_history(self, project: str = None, session_id: str = None, 
                                  limit: int = 50) -> List[Dict]:
        """Filter conversation history by project context"""
        with self.get_db_connection() as conn:
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query based on filters
            query = '''
                SELECT id, project, user_input, response_data, created_at, context_project
                FROM chat_threads 
                WHERE 1=1
            '''
            params = []
            
            if project and project != 'Main':
                query += ' AND (project = %s OR context_project = %s)'
                params.extend([project, project])
            
            query += ' ORDER BY created_at DESC LIMIT %s'
            params.append(limit)
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def route_integration_request(self, integration_type: str, request_data: Dict) -> Dict:
        """Route integration requests based on current project context"""
        current_context = self.get_project_context(request_data.get('session_id', ''))
        project = current_context['project']
        
        # Get project-specific mappings
        mappings = self.get_project_mappings(project)
        project_mappings = mappings.get(project, {})
        
        # Route based on integration type and current project
        routing_result = {
            'project': project,
            'integration_type': integration_type,
            'filtered_data': request_data,
            'project_mappings': project_mappings,
            'should_filter': project != 'Main'
        }
        
        # Analytics routing
        if integration_type == 'analytics':
            analytics_mappings = project_mappings.get('analytics', [])
            if analytics_mappings and project != 'Main':
                routing_result['filtered_data']['target_properties'] = [
                    mapping['identifier'] for mapping in analytics_mappings
                ]
        
        # Social media routing
        elif integration_type == 'social':
            social_mappings = project_mappings.get('social', [])
            if social_mappings and project != 'Main':
                routing_result['filtered_data']['target_accounts'] = [
                    mapping['identifier'] for mapping in social_mappings
                ]
        
        # Website/SEO routing
        elif integration_type == 'website':
            website_mappings = project_mappings.get('websites', [])
            if website_mappings and project != 'Main':
                routing_result['filtered_data']['target_domains'] = [
                    mapping['identifier'] for mapping in website_mappings
                ]
        
        return routing_result
    
    def get_project_dashboard_data(self, project: str) -> Dict:
        """Get comprehensive dashboard data for a project"""
        mappings = self.get_project_mappings(project)
        project_data = mappings.get(project, {})
        
        # Get recent conversation history
        recent_conversations = self.filter_conversation_history(project, limit=10)
        
        # Build dashboard data
        dashboard = {
            'project_name': project,
            'mappings': project_data,
            'recent_conversations': len(recent_conversations),
            'websites': len(project_data.get('websites', [])),
            'social_accounts': len(project_data.get('social', [])),
            'analytics_properties': len(project_data.get('analytics', [])),
            'email_accounts': len(project_data.get('email', [])),
            'last_activity': recent_conversations[0]['created_at'].isoformat() if recent_conversations else None
        }
        
        return dashboard
    
    def _refresh_mapping_cache(self):
        """Refresh the in-memory mapping cache"""
        self.mapping_cache = self.get_project_mappings()

# Usage example and integration points
def integrate_with_ghostline(app, project_mapping_system):
    """Integration points with main Ghostline app"""
    
    @app.route('/api/projects/mappings', methods=['GET'])
    def get_all_project_mappings():
        """API endpoint to get all project mappings"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        mappings = project_mapping_system.get_project_mappings()
        return jsonify(mappings)
    
    @app.route('/api/projects/context/<session_id>', methods=['GET', 'POST'])
    def manage_project_context(session_id):
        """API endpoint to get/set project context"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        if request.method == 'GET':
            context = project_mapping_system.get_project_context(session_id)
            return jsonify(context)
        
        elif request.method == 'POST':
            data = request.get_json()
            project = data.get('project')
            context_data = data.get('context_data', {})
            
            project_mapping_system.set_project_context(session_id, project, context_data)
            return jsonify({'success': True, 'project': project})
    
    @app.route('/api/projects/<project>/dashboard')
    def get_project_dashboard(project):
        """API endpoint for project dashboard data"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        dashboard_data = project_mapping_system.get_project_dashboard_data(project)
        return jsonify(dashboard_data)

if __name__ == '__main__':
    # Initialize the system
    mapping_system = ProjectMappingSystem()
    
    # Example usage
    print("🎯 Project Mapping System initialized")
    print("📊 Available projects:", mapping_system.CORE_PROJECTS)
    
    # Test mapping
    test_mappings = mapping_system.get_project_mappings()
    print("📂 Current mappings:", json.dumps(test_mappings, indent=2))