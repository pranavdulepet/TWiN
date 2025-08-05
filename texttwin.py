#!/usr/bin/env python3
"""
Clean TextTwin inference system
"""

import re
import json
import requests
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scripts.message_extractor import extract_conversation
from simple_rag import SimpleRAG

console = Console()

class TextTwin:
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.normalized = re.sub(r'[^\d]', '', phone_number)
        self.model_name = f"texttwin-{self.normalized}"
        self.base_model = "llama3.2:3b"
        self.messages = []
        
        # Load conversation history
        self._load_conversation()
        
        # Initialize RAG system
        self.rag = None
        self._init_rag()
        
    def _load_conversation(self):
        """Load conversation history"""
        console.print(f"📱 Loading conversation with {self.phone_number}...")
        
        # Try to load from existing file first
        conv_file = f"data/conversation_{self.normalized}.json"
        
        if Path(conv_file).exists():
            with open(conv_file, 'r') as f:
                self.messages = json.load(f)
                console.print(f"📂 Loaded {len(self.messages)} messages from cache")
        else:
            # Extract fresh messages
            self.messages = extract_conversation(self.phone_number)
            
        if not self.messages:
            console.print("❌ No conversation history found")
    
    def _init_rag(self):
        """Initialize RAG system"""
        if self.messages:
            try:
                console.print("🧠 Initializing conversation memory system...")
                self.rag = SimpleRAG(self.phone_number)
            except Exception as e:
                console.print(f"⚠️  Failed to initialize RAG: {e}")
                self.rag = None
    
    def _check_fine_tuned_model(self) -> bool:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                # handle ":latest" suffix
                return any(n.split(":")[0] == self.model_name for n in names)
        except Exception:
            pass
        return False

    def _get_context(self, recent_count: int = 10) -> str:
        """Get recent conversation context"""
        if not self.messages:
            return "No conversation history available."
        
        recent = self.messages[-recent_count:]
        context_parts = []
        
        for msg in recent:
            sender = "You" if msg['is_from_me'] else "Them"
            context_parts.append(f"{sender}: {msg['text']}")
        
        return "Recent conversation:\n" + "\n".join(context_parts)
    
    def generate_response(self, their_message: str) -> dict:
        """Generate response to their message"""
        
        has_fine_tuned = self._check_fine_tuned_model()
        model_to_use = self.model_name if has_fine_tuned else self.base_model
        
        console.print(f"🤖 Using model: {model_to_use}")
        
        if has_fine_tuned:
            # Use fine-tuned model directly
            prompt = their_message
        else:
            # Use context-based prompting
            context = self._get_context()
            your_messages = [m['text'] for m in self.messages if m['is_from_me']]
            
            prompt = f"""Based on this conversation history, respond to their message in my natural texting style:

{context}

Their message: {their_message}

Respond as I would naturally text this person. Be authentic to my style shown above."""
        
        try:
            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                return {
                    'response': generated_text,
                    'model': model_to_use,
                    'is_fine_tuned': has_fine_tuned,
                    'context_messages': len(self.messages)
                }
            else:
                return {'error': f"Ollama error: {response.status_code}"}
                
        except Exception as e:
            return {'error': f"Connection error: {e}"}
    
    def ask_question(self, question: str) -> dict:
        """Ask a question about the conversation history"""
        if not self.rag:
            return {'error': 'RAG system not available'}
        
        console.print(f"🔍 Searching conversation history...")
        result = self.rag.answer_question(question)
        return result
    
    def search_conversation(self, query: str, top_k: int = 5) -> dict:
        """Search conversation history"""
        if not self.rag:
            return {'error': 'RAG system not available'}
        
        results = self.rag.search_semantic(query, top_k=top_k)
        
        return {
            'query': query,
            'results': [
                {
                    'text': chunk.text,
                    'date': chunk.date,
                    'relevance_score': 0.8  # SimpleRAG doesn't return scores
                }
                for chunk in results
            ]
        }
    
    def get_conversation_stats(self) -> dict:
        """Get conversation statistics"""
        if not self.rag:
            return {'error': 'RAG system not available'}
        
        return self.rag.get_stats()
    
    def _handle_command(self, command: str):
        """Handle special commands"""
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        
        if cmd == '/help':
            self._show_help()
        elif cmd == '/ask':
            if len(parts) < 2:
                console.print("❌ Usage: /ask <question>")
                return
            question = parts[1]
            result = self.ask_question(question)
            self._display_answer(result)
        elif cmd == '/search':
            if len(parts) < 2:
                console.print("❌ Usage: /search <query>")
                return
            query = parts[1]
            result = self.search_conversation(query)
            self._display_search_results(result)
        elif cmd == '/stats':
            result = self.get_conversation_stats()
            self._display_stats(result)
        else:
            console.print(f"❌ Unknown command: {cmd}. Type /help for available commands.")
    
    def _show_help(self):
        """Show help information"""
        console.print(Panel(
            f"[bold cyan]TextTwin Commands[/bold cyan]\n\n"
            f"[bold]Text Generation:[/bold]\n"
            f"• Just type a message → Get a response in your style\n\n"
            f"[bold]Conversation Memory:[/bold]\n"
            f"• /ask <question> → Ask about your conversation history\n"
            f"• /search <query> → Search for specific conversations\n"
            f"• /stats → Show conversation statistics\n"
            f"[bold]General:[/bold]\n"
            f"• /help → Show this help\n"
            f"• quit → Exit",
            style="blue"
        ))
    
    def _display_answer(self, result: dict):
        """Display Q&A result"""
        if 'error' in result:
            console.print(f"❌ {result['error']}")
            return
        
        console.print(f"\n🎯 [bold green]Answer:[/bold green] {result['answer']}")
        console.print(f"📊 Confidence: {result['confidence']:.3f}")
        
        if result.get('sources'):
            console.print(f"\n📚 [bold]Sources ({len(result['sources'])} found):[/bold]")
            for i, source in enumerate(result['sources'][:3], 1):
                console.print(f"{i}. [{source['date']}] {source['text']}")
                console.print(f"   Relevance: {source['relevance_score']:.3f}")
    
    def _display_search_results(self, result: dict):
        """Display search results"""
        if 'error' in result:
            console.print(f"❌ {result['error']}")
            return
        
        results = result['results']
        console.print(f"\n🔍 [bold]Search Results for: '{result['query']}'[/bold]")
        console.print(f"Found {len(results)} matches\n")
        
        for i, res in enumerate(results, 1):
            console.print(f"[bold]{i}. [{res['date']}] (Score: {res['relevance_score']:.3f})[/bold]")
            console.print(f"{res['text']}\n")
    
    def _display_stats(self, stats: dict):
        """Display conversation statistics"""
        if 'error' in stats:
            console.print(f"❌ {stats['error']}")
            return
        
        table = Table(title="📊 Conversation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Phone Number", stats['phone_number'])
        table.add_row("Total Messages", f"{stats['total_messages']:,}")
        table.add_row("Your Messages", f"{stats['my_messages']:,}")
        table.add_row("Their Messages", f"{stats['their_messages']:,}")
        table.add_row("First Message", stats['first_message'])
        table.add_row("Last Message", stats['last_message'])
        table.add_row("Searchable Chunks", f"{stats['total_chunks']:,}")
        
        console.print(table)
    
    def interactive_mode(self):
        """Interactive chat mode"""
        
        rag_status = "✅ Available" if self.rag else "❌ Not available"
        
        console.print(Panel(
            f"🤖 [bold cyan]TextTwin Interactive Mode[/bold cyan]\n\n"
            f"Phone: {self.phone_number}\n"
            f"Messages in history: {len(self.messages)}\n"
            f"Fine-tuned model: {'✅ Available' if self._check_fine_tuned_model() else '❌ Not found'}\n"
            f"Conversation Memory: {rag_status}\n\n"
            f"[bold]Commands:[/bold]\n"
            f"• Type messages → Get responses in your style\n"
            f"• /ask <question> → Ask about conversation history\n"
            f"• /search <query> → Search conversations\n"
            f"• /stats → Show conversation statistics\n"
            f"• /help → Show this help\n"
            f"• quit → Exit\n",
            style="cyan"
        ))
        
        while True:
            try:
                user_input = input("\n💬 Input: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # Regular message - generate response
                    console.print("🤔 Generating response...")
                    result = self.generate_response(user_input)
                    
                    if 'error' in result:
                        console.print(f"❌ {result['error']}")
                    else:
                        console.print(f"🤖 Your response: [bold green]{result['response']}[/bold green]")
                        console.print(f"   Model: {result['model']} | Messages: {result['context_messages']}")
                
            except KeyboardInterrupt:
                break
        
        console.print("\n👋 Goodbye!")

def main():
    import sys
    
    if len(sys.argv) != 2:
        console.print("Usage: python3 texttwin.py <phone_number>")
        console.print("Example: python3 texttwin.py 'XXXXXXXXXX'")
        sys.exit(1)
    
    phone = sys.argv[1]
    
    console.print("🤖 [bold cyan]TextTwin - Personal Writing Style Cloner[/bold cyan]")
    console.print("=" * 55)
    
    twin = TextTwin(phone)
    twin.interactive_mode()

if __name__ == "__main__":
    main()