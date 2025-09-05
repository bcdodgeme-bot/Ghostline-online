# modules/cloze_clickup_integration.py - Production-ready integration
"""
Cloze + ClickUp Integration Module
Connects relationship intelligence with task management for productivity optimization
"""

import os
import datetime
import json
from typing import Dict, List, Optional, Any
from modules.cloze_integration import ClozeClient, is_cloze_configured
from modules.clickup_integration import ClickUpClient, is_clickup_configured
from modules.database import save_conversation_enhanced
from utils.ghostline_engine import generate_response

class ClozeClickUpIntegration:
    def __init__(self):
        self.cloze_client = ClozeClient() if is_cloze_configured() else None
        self.clickup_client = ClickUpClient() if is_clickup_configured() else None
        
        # Default configurations from your test results
        self.default_list_id = "901306635049"  # Personal Time Management → List
        self.default_team_id = "9013453647"    # Rose and Angel Consulting
        
        if not self.cloze_client or not self.clickup_client:
            raise ValueError("Both Cloze and ClickUp must be configured for integration")
    
    def analyze_relationship_priorities(self, limit: int = 50) -> Dict[str, Any]:
        """Analyze Cloze relationship data to identify follow-up priorities"""
        try:
            # Get people with stages from Cloze
            people_data = self.cloze_client.get_people_stages(limit=limit)
            
            if not people_data or not people_data.get('data'):
                return {"error": "No people data available from Cloze"}
            
            people = people_data['data']
            
            # Analyze relationship priorities
            priority_analysis = {
                'high_priority': [],
                'medium_priority': [],
                'follow_up_needed': [],
                'total_people': len(people)
            }
            
            for person in people:
                person_info = {
                    'name': person.get('name', 'Unknown'),
                    'stage': person.get('stage', 'No Stage'),
                    'company': person.get('company', ''),
                    'email': person.get('email', ''),
                    'last_contact': person.get('lastContact', ''),
                    'id': person.get('id', '')
                }
                
                # Prioritize based on stage and contact history
                stage = person_info['stage'].lower()
                
                if any(keyword in stage for keyword in ['qualified', 'hot', 'proposal', 'closing']):
                    priority_analysis['high_priority'].append(person_info)
                elif any(keyword in stage for keyword in ['lead', 'interested', 'warm']):
                    priority_analysis['medium_priority'].append(person_info)
                elif any(keyword in stage for keyword in ['cold', 'follow', 'nurture']):
                    priority_analysis['follow_up_needed'].append(person_info)
            
            return priority_analysis
            
        except Exception as e:
            return {"error": f"Relationship analysis failed: {str(e)}"}
    
    def get_email_engagement_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Get recent email engagement data from Cloze"""
        try:
            engagement_data = self.cloze_client.get_message_opens(
                days_back=days_back, 
                limit=50
            )
            
            if not engagement_data or not engagement_data.get('data'):
                return {"engaged_contacts": [], "total_opens": 0}
            
            opens = engagement_data['data']
            engaged_contacts = []
            
            for open_event in opens:
                contact_info = {
                    'person_name': open_event.get('person', {}).get('name', 'Unknown'),
                    'subject': open_event.get('subject', 'No Subject')[:50],
                    'opened_at': open_event.get('opened_at', ''),
                    'person_id': open_event.get('person', {}).get('id', '')
                }
                engaged_contacts.append(contact_info)
            
            return {
                "engaged_contacts": engaged_contacts,
                "total_opens": len(opens),
                "analysis_period": f"Last {days_back} days"
            }
            
        except Exception as e:
            return {"error": f"Email engagement analysis failed: {str(e)}"}
    
    def create_relationship_tasks(self, priority_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create ClickUp tasks based on relationship priorities"""
        try:
            tasks_created = []
            
            # High priority contacts - immediate action needed
            for person in priority_analysis.get('high_priority', [])[:5]:  # Limit to top 5
                task_name = f"URGENT: Follow up with {person['name']}"
                description = self._build_task_description(person, "HIGH PRIORITY CONTACT")
                
                task = self.clickup_client.create_task(
                    name=task_name,
                    description=description,
                    priority=1,  # Urgent
                    list_id=self.default_list_id
                )
                
                tasks_created.append({
                    "type": "high_priority",
                    "person": person['name'],
                    "task_id": task.get('id'),
                    "task_url": task.get('url')
                })
            
            # Medium priority contacts - follow up this week
            for person in priority_analysis.get('medium_priority', [])[:3]:  # Limit to top 3
                task_name = f"Follow up with {person['name']} - {person['stage']}"
                description = self._build_task_description(person, "MEDIUM PRIORITY CONTACT")
                
                # Due date: 3 days from now
                due_date = datetime.datetime.now() + datetime.timedelta(days=3)
                
                task = self.clickup_client.create_task(
                    name=task_name,
                    description=description,
                    priority=2,  # High
                    due_date=due_date,
                    list_id=self.default_list_id
                )
                
                tasks_created.append({
                    "type": "medium_priority",
                    "person": person['name'],
                    "task_id": task.get('id'),
                    "task_url": task.get('url')
                })
            
            # Follow-up needed contacts - nurture campaign
            follow_up_count = len(priority_analysis.get('follow_up_needed', []))
            if follow_up_count > 0:
                task_name = f"Plan nurture campaign for {follow_up_count} contacts"
                description = f"Create nurture sequence for contacts needing follow-up:\n\n"
                
                for person in priority_analysis.get('follow_up_needed', [])[:10]:
                    description += f"• {person['name']} ({person['company']}) - {person['stage']}\n"
                
                # Due date: 1 week from now
                due_date = datetime.datetime.now() + datetime.timedelta(days=7)
                
                task = self.clickup_client.create_task(
                    name=task_name,
                    description=description,
                    priority=3,  # Normal
                    due_date=due_date,
                    list_id=self.default_list_id
                )
                
                tasks_created.append({
                    "type": "nurture_campaign",
                    "contact_count": follow_up_count,
                    "task_id": task.get('id'),
                    "task_url": task.get('url')
                })
            
            return {
                "success": True,
                "tasks_created": len(tasks_created),
                "details": tasks_created
            }
            
        except Exception as e:
            return {"error": f"Task creation failed: {str(e)}"}
    
    def create_engagement_tasks(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create ClickUp tasks for engaged email contacts"""
        try:
            if not engagement_data.get('engaged_contacts'):
                return {"message": "No email engagement to process"}
            
            tasks_created = []
            engaged_contacts = engagement_data['engaged_contacts']
            
            # Group by person (in case someone opened multiple emails)
            person_engagement = {}
            for contact in engaged_contacts:
                person_name = contact['person_name']
                if person_name not in person_engagement:
                    person_engagement[person_name] = []
                person_engagement[person_name].append(contact)
            
            # Create tasks for most engaged contacts
            for person_name, engagements in list(person_engagement.items())[:5]:  # Top 5 engaged
                task_name = f"Follow up with engaged contact: {person_name}"
                
                description = f"Recent email engagement from {person_name}:\n\n"
                for engagement in engagements:
                    description += f"• Opened: {engagement['subject']}\n"
                
                description += f"\nTotal opens: {len(engagements)}\n"
                description += f"Engagement level: {'High' if len(engagements) > 2 else 'Medium'}\n"
                description += "\nSuggested action: Personal follow-up call or email"
                
                # Due date: 1 day from now for engaged contacts
                due_date = datetime.datetime.now() + datetime.timedelta(days=1)
                
                task = self.clickup_client.create_task(
                    name=task_name,
                    description=description,
                    priority=2,  # High priority for engaged contacts
                    due_date=due_date,
                    list_id=self.default_list_id
                )
                
                tasks_created.append({
                    "type": "engagement_follow_up",
                    "person": person_name,
                    "engagement_count": len(engagements),
                    "task_id": task.get('id'),
                    "task_url": task.get('url')
                })
            
            return {
                "success": True,
                "tasks_created": len(tasks_created),
                "details": tasks_created
            }
            
        except Exception as e:
            return {"error": f"Engagement task creation failed: {str(e)}"}
    
    def _build_task_description(self, person: Dict[str, Any], priority_level: str) -> str:
        """Build detailed task description with person context"""
        description = f"{priority_level}\n\n"
        description += f"**Contact:** {person['name']}\n"
        
        if person.get('company'):
            description += f"**Company:** {person['company']}\n"
        
        description += f"**Stage:** {person['stage']}\n"
        
        if person.get('email'):
            description += f"**Email:** {person['email']}\n"
        
        if person.get('last_contact'):
            description += f"**Last Contact:** {person['last_contact']}\n"
        
        description += "\n**Suggested Actions:**\n"
        description += "• Review recent communication history\n"
        description += "• Prepare personalized follow-up\n"
        description += "• Schedule meeting if appropriate\n"
        description += "• Update Cloze with interaction notes\n"
        
        description += f"\n**Created by:** Ghostline Cloze Integration\n"
        description += f"**Created:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        return description
    
    def generate_productivity_briefing(self) -> Dict[str, Any]:
        """Generate comprehensive productivity briefing combining both systems"""
        try:
            briefing_data = {}
            
            # Get relationship priorities
            relationship_analysis = self.analyze_relationship_priorities()
            briefing_data['relationship_analysis'] = relationship_analysis
            
            # Get email engagement
            engagement_data = self.get_email_engagement_data()
            briefing_data['engagement_data'] = engagement_data
            
            # Get ClickUp task summary
            today = datetime.datetime.now()
            week_start = today - datetime.timedelta(days=today.weekday())
            
            try:
                time_entries = self.clickup_client.get_time_entries(
                    team_id=self.default_team_id,
                    start_date=week_start,
                    end_date=today
                )
                
                total_time = sum(int(entry.get('duration', 0)) for entry in time_entries.get('data', []))
                hours_worked = total_time / (1000 * 60 * 60)  # Convert ms to hours
                
                briefing_data['productivity_metrics'] = {
                    "hours_worked_this_week": round(hours_worked, 2),
                    "time_entries": len(time_entries.get('data', [])),
                    "average_daily_hours": round(hours_worked / 7, 2)
                }
            except Exception as e:
                briefing_data['productivity_metrics'] = {"error": str(e)}
            
            # Generate AI summary
            summary_prompt = self._build_briefing_prompt(briefing_data)
            
            return {
                "success": True,
                "raw_data": briefing_data,
                "summary_prompt": summary_prompt
            }
            
        except Exception as e:
            return {"error": f"Briefing generation failed: {str(e)}"}
    
    def _build_briefing_prompt(self, briefing_data: Dict[str, Any]) -> str:
        """Build AI prompt for productivity briefing"""
        prompt = "Generate a productivity briefing based on this Cloze + ClickUp data:\n\n"
        
        # Relationship data
        rel_data = briefing_data.get('relationship_analysis', {})
        if 'error' not in rel_data:
            prompt += f"RELATIONSHIP PRIORITIES:\n"
            prompt += f"• High priority contacts: {len(rel_data.get('high_priority', []))}\n"
            prompt += f"• Medium priority contacts: {len(rel_data.get('medium_priority', []))}\n"
            prompt += f"• Follow-up needed: {len(rel_data.get('follow_up_needed', []))}\n"
            prompt += f"• Total people in pipeline: {rel_data.get('total_people', 0)}\n\n"
        
        # Engagement data
        eng_data = briefing_data.get('engagement_data', {})
        if 'error' not in eng_data:
            prompt += f"EMAIL ENGAGEMENT (Last 7 days):\n"
            prompt += f"• Total email opens: {eng_data.get('total_opens', 0)}\n"
            prompt += f"• Engaged contacts: {len(eng_data.get('engaged_contacts', []))}\n\n"
        
        # Productivity metrics
        prod_data = briefing_data.get('productivity_metrics', {})
        if 'error' not in prod_data:
            prompt += f"PRODUCTIVITY METRICS:\n"
            prompt += f"• Hours worked this week: {prod_data.get('hours_worked_this_week', 0)}\n"
            prompt += f"• Time entries logged: {prod_data.get('time_entries', 0)}\n"
            prompt += f"• Average daily hours: {prod_data.get('average_daily_hours', 0)}\n\n"
        
        prompt += "Provide:\n"
        prompt += "1. Top 3 relationship priorities for today\n"
        prompt += "2. Recommended time allocation\n"
        prompt += "3. Key productivity insights\n"
        prompt += "4. Specific action items\n"
        
        return prompt

def process_cloze_clickup_command(user_input: str, project: str, use_voices: List[str], random_toggle: bool) -> tuple:
    """Process Cloze + ClickUp integration commands"""
    user_lower = user_input.lower().strip()
    
    # Check if both systems are configured
    if not (is_cloze_configured() and is_clickup_configured()):
        return {
            "SyntaxPrime": "Both Cloze and ClickUp must be configured for integration features.\n" +
                          "Visit /integrations to set up missing connections."
        }, True
    
    try:
        integration = ClozeClickUpIntegration()
        
        # Relationship analysis command
        if any(phrase in user_lower for phrase in [
            'relationship priorities', 'cloze priorities', 'relationship analysis',
            'who should i follow up with', 'priority contacts'
        ]):
            analysis = integration.analyze_relationship_priorities()
            
            if 'error' in analysis:
                response = f"**Relationship Analysis Error:**\n{analysis['error']}"
            else:
                response = "**🎯 RELATIONSHIP PRIORITIES ANALYSIS**\n\n"
                response += f"**📊 Pipeline Overview:**\n"
                response += f"• Total contacts: {analysis['total_people']}\n"
                response += f"• High priority: {len(analysis['high_priority'])}\n"
                response += f"• Medium priority: {len(analysis['medium_priority'])}\n"
                response += f"• Follow-up needed: {len(analysis['follow_up_needed'])}\n\n"
                
                if analysis['high_priority']:
                    response += "**🔥 HIGH PRIORITY (Immediate Action):**\n"
                    for person in analysis['high_priority'][:5]:
                        response += f"• **{person['name']}** ({person['company']}) - {person['stage']}\n"
                    response += "\n"
                
                if analysis['medium_priority']:
                    response += "**⚡ MEDIUM PRIORITY (This Week):**\n"
                    for person in analysis['medium_priority'][:3]:
                        response += f"• **{person['name']}** ({person['company']}) - {person['stage']}\n"
                    response += "\n"
                
                response += "Use `create relationship tasks` to auto-generate ClickUp tasks for these contacts."
            
            return {"SyntaxPrime": response}, True
        
        # Task creation command
        elif any(phrase in user_lower for phrase in [
            'create relationship tasks', 'create priority tasks', 'make tasks from cloze',
            'generate relationship tasks'
        ]):
            # Get relationship analysis
            analysis = integration.analyze_relationship_priorities()
            
            if 'error' in analysis:
                response = f"**Task Creation Failed:**\n{analysis['error']}"
            else:
                # Create tasks
                task_result = integration.create_relationship_tasks(analysis)
                
                if task_result.get('success'):
                    response = f"**✅ RELATIONSHIP TASKS CREATED**\n\n"
                    response += f"**Created {task_result['tasks_created']} tasks in ClickUp:**\n\n"
                    
                    for task in task_result['details']:
                        if task['type'] == 'high_priority':
                            response += f"🔥 **URGENT:** {task['person']}\n"
                        elif task['type'] == 'medium_priority':
                            response += f"⚡ **Follow-up:** {task['person']}\n"
                        elif task['type'] == 'nurture_campaign':
                            response += f"📧 **Nurture Campaign:** {task['contact_count']} contacts\n"
                        
                        if task.get('task_url'):
                            response += f"   🔗 [View Task]({task['task_url']})\n"
                        response += "\n"
                    
                    response += "All tasks created in your **Personal Time Management** list."
                else:
                    response = f"**Task Creation Failed:**\n{task_result.get('error', 'Unknown error')}"
            
            return {"SyntaxPrime": response}, True
        
        # Email engagement command
        elif any(phrase in user_lower for phrase in [
            'email engagement', 'who opened emails', 'engagement analysis',
            'email opens', 'engaged contacts'
        ]):
            engagement = integration.get_email_engagement_data()
            
            if 'error' in engagement:
                response = f"**Email Engagement Error:**\n{engagement['error']}"
            else:
                response = "**📧 EMAIL ENGAGEMENT ANALYSIS**\n\n"
                response += f"**📊 Summary ({engagement['analysis_period']}):**\n"
                response += f"• Total email opens: {engagement['total_opens']}\n"
                response += f"• Engaged contacts: {len(engagement['engaged_contacts'])}\n\n"
                
                if engagement['engaged_contacts']:
                    response += "**🔥 MOST ENGAGED CONTACTS:**\n"
                    for contact in engagement['engaged_contacts'][:5]:
                        response += f"• **{contact['person_name']}** - {contact['subject']}\n"
                    response += "\n"
                    response += "Use `create engagement tasks` to follow up with engaged contacts."
                else:
                    response += "No recent email engagement detected."
            
            return {"SyntaxPrime": response}, True
        
        # Engagement task creation
        elif any(phrase in user_lower for phrase in [
            'create engagement tasks', 'follow up engaged contacts',
            'make tasks from engagement'
        ]):
            # Get engagement data
            engagement = integration.get_email_engagement_data()
            
            if 'error' in engagement:
                response = f"**Engagement Task Creation Failed:**\n{engagement['error']}"
            else:
                # Create engagement tasks
                task_result = integration.create_engagement_tasks(engagement)
                
                if task_result.get('success'):
                    response = f"**✅ ENGAGEMENT TASKS CREATED**\n\n"
                    response += f"**Created {task_result['tasks_created']} follow-up tasks:**\n\n"
                    
                    for task in task_result['details']:
                        response += f"📧 **{task['person']}** ({task['engagement_count']} opens)\n"
                        if task.get('task_url'):
                            response += f"   🔗 [View Task]({task['task_url']})\n"
                        response += "\n"
                    
                    response += "All tasks created with 1-day due dates for immediate follow-up."
                else:
                    response = f"**Task Creation Failed:**\n{task_result.get('error', 'Unknown error')}"
            
            return {"SyntaxPrime": response}, True
        
        # Productivity briefing command
        elif any(phrase in user_lower for phrase in [
            'productivity briefing', 'cloze clickup briefing', 'relationship productivity',
            'integration briefing', 'combined briefing'
        ]):
            briefing_result = integration.generate_productivity_briefing()
            
            if briefing_result.get('success'):
                # Generate AI response for the briefing
                try:
                    from utils.ghostline_engine import generate_response
                    
                    ai_response = generate_response(
                        briefing_result['summary_prompt'],
                        use_voices,
                        random_toggle,
                        project=project,
                        model=os.getenv("CHAT_MODEL", "openrouter/auto"),
                        retrieval_context=[]
                    )
                    
                    # Save the interaction
                    save_conversation_enhanced(project, user_input, ai_response)
                    
                    return ai_response, True
                    
                except Exception as e:
                    return {
                        "SyntaxPrime": f"Briefing generation failed: {str(e)}\n\n" +
                                      f"Raw data available - try `relationship priorities` for analysis."
                    }, True
            else:
                return {
                    "SyntaxPrime": f"**Productivity Briefing Failed:**\n{briefing_result.get('error', 'Unknown error')}"
                }, True
        
        # No matching command
        return {}, False
        
    except Exception as e:
        return {
            "SyntaxPrime": f"**Cloze + ClickUp Integration Error:**\n{str(e)}\n\n" +
                          "Check that both integrations are properly configured."
        }, True

# Export the main integration class and command processor
__all__ = ['ClozeClickUpIntegration', 'process_cloze_clickup_command']