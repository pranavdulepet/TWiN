#!/usr/bin/env python3
"""
Contact-Specific Message Generator - TextTwin Extension
======================================================

Generates messages tailored to specific contacts based on your conversation history with them.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

from texttwin_engine import TextTwinEngine
from safe_imessage_reader import SafeIMessageReader

console = Console()

class ContactSpecificGenerator(TextTwinEngine):
    """Extends TextTwin to generate messages for specific contacts."""
    
    def __init__(self):
        super().__init__()
        self.contact_conversations = {}
        self.contact_styles = {}
    
    def analyze_contact_conversations(self, contact_identifier: str = None) -> Dict:
        """
        Analyze conversations with specific contacts.
        
        Args:
            contact_identifier: Phone number, email, or contact name to filter by
        """
        try:
            with SafeIMessageReader() as reader:
                # Get all messages with chat information
                query = """
                SELECT 
                    m.text,
                    m.date,
                    m.is_from_me,
                    h.chat_identifier,
                    h.display_name
                FROM message m
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                LEFT JOIN chat h ON cmj.chat_id = h.ROWID
                WHERE m.text IS NOT NULL 
                AND m.text != ''
                AND h.chat_identifier IS NOT NULL
                ORDER BY m.date DESC
                LIMIT 5000
                """
                
                df = pd.read_sql_query(query, reader.connection)
                
                # Group conversations by contact
                conversations = defaultdict(list)
                
                for _, row in df.iterrows():
                    contact_id = row['chat_identifier']
                    
                    # Filter by specific contact if provided
                    if contact_identifier and contact_identifier not in str(contact_id):
                        continue
                    
                    conversations[contact_id].append({
                        'text': row['text'],
                        'is_from_me': bool(row['is_from_me']),
                        'timestamp': row['date'],
                        'display_name': row['display_name']
                    })
                
                self.contact_conversations = dict(conversations)
                console.print(f"✓ Analyzed conversations with {len(conversations)} contacts")
                
                return self.contact_conversations
                
        except Exception as e:
            console.print(f"[red]Could not analyze contact conversations: {e}[/red]")
            # Fall back to demo data
            self._create_demo_contact_data()
            return self.contact_conversations
    
    def _create_demo_contact_data(self):
        """Create demo contact conversation data."""
        self.contact_conversations = {
            "+1234567890": [
                {'text': "hey want to grab lunch?", 'is_from_me': True, 'timestamp': '2024-01-01 12:00:00'},
                {'text': "sure! where?", 'is_from_me': False, 'timestamp': '2024-01-01 12:01:00'},
                {'text': "that new pizza place?", 'is_from_me': True, 'timestamp': '2024-01-01 12:02:00'},
                {'text': "perfect see you there", 'is_from_me': False, 'timestamp': '2024-01-01 12:03:00'},
            ],
            "work_group": [
                {'text': "meeting at 3pm today", 'is_from_me': True, 'timestamp': '2024-01-01 10:00:00'},
                {'text': "got it, thanks", 'is_from_me': False, 'timestamp': '2024-01-01 10:01:00'},
                {'text': "ill send the agenda", 'is_from_me': True, 'timestamp': '2024-01-01 10:02:00'},
            ],
            "mom": [
                {'text': "hi honey how are you?", 'is_from_me': False, 'timestamp': '2024-01-01 09:00:00'},
                {'text': "good mom! just busy with work", 'is_from_me': True, 'timestamp': '2024-01-01 09:05:00'},
                {'text': "make sure you eat well", 'is_from_me': False, 'timestamp': '2024-01-01 09:06:00'},
                {'text': "will do ❤️", 'is_from_me': True, 'timestamp': '2024-01-01 09:07:00'},
            ]
        }
        console.print("✓ Created demo contact conversation data")
    
    def analyze_contact_specific_style(self, contact_id: str) -> Optional[Dict]:
        """
        Analyze how you communicate with a specific contact.
        
        Args:
            contact_id: The contact identifier to analyze
            
        Returns:
            Dictionary with contact-specific communication patterns
        """
        if contact_id not in self.contact_conversations:
            console.print(f"[red]No conversation data found for {contact_id}[/red]")
            return None
        
        messages = self.contact_conversations[contact_id]
        your_messages = [msg for msg in messages if msg['is_from_me']]
        their_messages = [msg for msg in messages if not msg['is_from_me']]
        
        if not your_messages:
            console.print(f"[red]No messages from you found for {contact_id}[/red]")
            return None
        
        # Analyze your style with this specific contact
        your_texts = [msg['text'] for msg in your_messages]
        their_texts = [msg['text'] for msg in their_messages]
        
        # Basic metrics
        avg_your_length = sum(len(text) for text in your_texts) / len(your_texts)
        avg_their_length = sum(len(text) for text in their_texts) / len(their_texts) if their_texts else 0
        
        # Response patterns
        response_ratio = len(your_messages) / len(messages) if messages else 0
        
        # Common topics/words in this conversation
        all_your_words = ' '.join(your_texts).lower().split()
        word_freq = defaultdict(int)
        for word in all_your_words:
            if len(word) > 3:  # Only count substantial words
                word_freq[word] += 1
        
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        style_profile = {
            'contact_id': contact_id,
            'message_count': len(your_messages),
            'avg_message_length': avg_your_length,
            'response_ratio': response_ratio,
            'common_words': common_words,
            'formality_level': self._assess_formality(your_texts),
            'recent_messages': your_texts[-5:],  # Last 5 messages for context
        }
        
        self.contact_styles[contact_id] = style_profile
        return style_profile
    
    def _assess_formality(self, texts: List[str]) -> str:
        """Assess the formality level of messages."""
        formal_indicators = ['please', 'thank you', 'regarding', 'however', 'therefore']
        casual_indicators = ['hey', 'lol', 'omg', 'btw', 'gonna', 'wanna']
        
        formal_count = sum(1 for text in texts for indicator in formal_indicators if indicator in text.lower())
        casual_count = sum(1 for text in texts for indicator in casual_indicators if indicator in text.lower())
        
        if formal_count > casual_count:
            return "formal"
        elif casual_count > formal_count:
            return "casual"
        else:
            return "neutral"
    
    def generate_contact_specific_response(self, contact_id: str, context_message: str, message_type: str = "response") -> str:
        """
        Generate a message tailored for a specific contact.
        
        Args:
            contact_id: The contact to generate a message for
            context_message: The message you're responding to (if any)
            message_type: 'response', 'initiate', or 'follow_up'
        """
        if contact_id not in self.contact_styles:
            console.print(f"[yellow]No specific style data for {contact_id}, analyzing now...[/yellow]")
            self.analyze_contact_specific_style(contact_id)
        
        contact_style = self.contact_styles.get(contact_id, {})
        
        # Create contact-specific prompt
        base_prompt = self.create_style_prompt()
        
        contact_context = f"""
SPECIFIC CONTACT CONTEXT:
- You're texting: {contact_id}
- Your message style with them: {contact_style.get('formality_level', 'neutral')}
- Your typical message length with them: {contact_style.get('avg_message_length', 30):.1f} characters
- Recent messages you sent them: {contact_style.get('recent_messages', [])}
- Common words you use with them: {[word for word, count in contact_style.get('common_words', [])[:5]]}

MESSAGE TYPE: {message_type}
"""
        
        if message_type == "response":
            full_prompt = f"{base_prompt}\n{contact_context}\nThey said: \"{context_message}\"\n\nRespond as you typically would to this contact:"
        elif message_type == "initiate":
            full_prompt = f"{base_prompt}\n{contact_context}\nContext: {context_message}\n\nStart a conversation with this contact about this topic:"
        else:  # follow_up
            full_prompt = f"{base_prompt}\n{contact_context}\nPrevious context: {context_message}\n\nSend a follow-up message to this contact:"
        
        return self._generate_with_ollama(full_prompt)
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama."""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 100
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip().strip('"\'')
            else:
                return "sorry, having trouble generating a response"
                
        except Exception as e:
            console.print(f"[red]Generation error: {e}[/red]")
            return "hmm something went wrong"
    
    def list_contacts(self) -> None:
        """Display available contacts."""
        if not self.contact_conversations:
            console.print("[red]No contact data available. Run analyze_contact_conversations() first.[/red]")
            return
        
        table = Table(title="📱 Your Contacts", style="cyan")
        table.add_column("Contact ID", style="bold")
        table.add_column("Message Count", justify="right")
        table.add_column("Your Messages", justify="right")
        table.add_column("Style Analysis", style="dim")
        
        for contact_id, messages in self.contact_conversations.items():
            your_msg_count = sum(1 for msg in messages if msg['is_from_me'])
            total_messages = len(messages)
            
            style_status = "✓ Analyzed" if contact_id in self.contact_styles else "Not analyzed"
            
            # Clean up contact ID for display
            display_id = contact_id[:20] + "..." if len(contact_id) > 20 else contact_id
            
            table.add_row(
                display_id,
                str(total_messages),
                str(your_msg_count),
                style_status
            )
        
        console.print(table)
    
    def contact_chat_demo(self, contact_id: str) -> None:
        """Interactive demo for a specific contact."""
        if contact_id not in self.contact_conversations:
            console.print(f"[red]Contact {contact_id} not found. Available contacts:[/red]")
            self.list_contacts()
            return
        
        # Analyze their style if not already done
        if contact_id not in self.contact_styles:
            self.analyze_contact_specific_style(contact_id)
        
        style = self.contact_styles[contact_id]
        
        console.print(Panel(
            f"💬 [bold cyan]Chatting with: {contact_id}[/bold cyan]\n\n" +
            f"📊 Your style with them:\n" +
            f"• Formality: {style['formality_level']}\n" +
            f"• Avg message length: {style['avg_message_length']:.1f} chars\n" +
            f"• Total messages: {style['message_count']}\n\n" +
            "[dim]Type messages and see how you'd typically respond to this contact[/dim]",
            style="blue"
        ))
        
        # Show recent conversation context
        recent_messages = self.contact_conversations[contact_id][-5:]
        console.print("\n[bold]Recent conversation:[/bold]")
        for msg in recent_messages:
            sender = "You" if msg['is_from_me'] else "Them"
            console.print(f"[dim]{sender}:[/dim] {msg['text']}")
        
        console.print("\n" + "="*50)
        
        # Demo responses
        test_scenarios = [
            ("response", "hey are you free this weekend?"),
            ("response", "thanks for helping me out"),
            ("initiate", "want to make dinner plans"),
            ("follow_up", "following up on our conversation about the project"),
        ]
        
        for msg_type, context in test_scenarios:
            console.print(f"\n[bold green]Scenario ({msg_type}):[/bold green] {context}")
            response = self.generate_contact_specific_response(contact_id, context, msg_type)
            console.print(f"[bold blue]You would text:[/bold blue] {response}")


def demo_contact_specific():
    """Demo the contact-specific message generation."""
    console.print(Panel(
        "📱 [bold magenta]Contact-Specific Message Generator[/bold magenta]\n\n" +
        "Generate messages tailored to specific contacts based on your conversation history!",
        style="magenta"
    ))
    
    generator = ContactSpecificGenerator()
    
    # First analyze your general style
    console.print("\n[bold cyan]Step 1: Analyzing your general texting style...[/bold cyan]")
    generator.analyze_sample_messages()
    
    # Then analyze contact-specific conversations
    console.print("\n[bold cyan]Step 2: Analyzing contact-specific conversations...[/bold cyan]")
    generator.analyze_contact_conversations()
    
    # Show available contacts
    console.print("\n[bold cyan]Step 3: Available contacts[/bold cyan]")
    generator.list_contacts()
    
    # Demo with each contact
    for contact_id in list(generator.contact_conversations.keys())[:2]:  # Limit to first 2 for demo
        console.print(f"\n[bold cyan]Step 4: Demo with {contact_id}[/bold cyan]")
        generator.contact_chat_demo(contact_id)


if __name__ == "__main__":
    demo_contact_specific()