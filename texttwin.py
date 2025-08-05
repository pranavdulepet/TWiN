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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from scripts.message_extractor import extract_conversation

# Try to import RAG system, fallback if dependencies are missing
try:
    from simple_rag import SimpleRAG
    RAG_AVAILABLE = True
except ImportError as e:
    console = Console()
    console.print(f"⚠️  RAG system unavailable due to dependency issue: {e}")
    console.print("📝 Running in basic mode without conversation memory features")
    console.print("💡 To fix: pip install --upgrade sentence-transformers huggingface_hub")
    SimpleRAG = None
    RAG_AVAILABLE = False

console = Console()

class TextTwin:
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.normalized = re.sub(r'[^\d]', '', phone_number)
        self.model_name = f"texttwin-{self.normalized}"
        self.base_model = "llama3.2:3b"
        self.messages = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            
            init_task = progress.add_task("🤖 Initializing TextTwin...", total=100)
            
            # Load conversation history
            progress.update(init_task, advance=50, description="📱 Loading conversation history...")
            self._load_conversation()
            
            # Initialize RAG system
            progress.update(init_task, advance=50, description="🧠 Initializing conversation memory...")
            self.rag = None
            self._init_rag()
            
            progress.update(init_task, completed=100, description="✅ TextTwin ready")
        
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
        if not RAG_AVAILABLE:
            console.print("⚠️  Conversation memory not available (dependency issues)")
            self.rag = None
            return
            
        if self.messages:
            try:
                console.print("🧠 Initializing conversation memory system...")
                self.rag = SimpleRAG(self.phone_number)
            except Exception as e:
                console.print(f"⚠️  Failed to initialize RAG: {e}")
                self.rag = None
        else:
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
    
    def _clean_generated_message(self, text: str) -> str:
        """Clean up generated message output"""
        import re
        
        # Remove common unwanted patterns
        text = text.strip()
        
        # Remove conversation formatting (You:, Them:, +, -, etc.)
        text = re.sub(r'^(You:|Them:|Me:|\+|\-|&|#|\*|!|\d+)', '', text, flags=re.MULTILINE)
        
        # Remove loved/reaction indicators
        text = re.sub(r'Loved\s+"[^"]*"', '', text, flags=re.MULTILINE)
        text = re.sub(r'Loved an image', '', text, flags=re.MULTILINE)
        text = re.sub(r'Gaf een hartje aan', '', text, flags=re.MULTILINE)
        
        # Remove metadata/formatting characters
        text = re.sub(r'￼+', '', text)  # Object replacement characters
        text = re.sub(r'[ALoved]+\s*"[^"]*"', '', text)
        
        # Split by lines and take only clean message lines
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, metadata, conversation history
            if (line and 
                not line.startswith('Context') and
                not line.startswith('Intent:') and
                not line.startswith('Instructions:') and
                not line.startswith('My message:') and
                not line.startswith('Recent conversation:') and
                not 'conversation history' in line.lower() and
                len(line) < 200):  # Reasonable message length
                clean_lines.append(line)
        
        # Take the first reasonable line as the message
        if clean_lines:
            message = clean_lines[0]
            # Clean up any remaining formatting
            message = re.sub(r'^[+\-&#*!]+\s*', '', message)
            message = re.sub(r'\s+', ' ', message)  # Normalize whitespace
            return message.strip()
        
        # Fallback: try to extract from original text
        fallback = re.sub(r'[+\-&#*!￼]+', '', text)
        fallback = re.sub(r'\s+', ' ', fallback)
        
        # Take first sentence if it's reasonable length
        sentences = fallback.split('.')
        if sentences and len(sentences[0].strip()) < 100 and len(sentences[0].strip()) > 0:
            return sentences[0].strip()
        
        # Last resort: return first 100 chars
        return fallback[:100].strip() if fallback.strip() else "Error: Could not generate clean message"
    
    def _generate_with_base_model(self, my_intent: str) -> dict:
        """Fallback to base model when fine-tuned model fails"""
        context = self._get_context(recent_count=5)  # Shorter context
        your_messages = [m['text'] for m in self.messages if m['is_from_me']]
        style_examples = "\n".join([f"- {msg}" for msg in your_messages[-5:]])  # Recent examples
        
        prompt = f"""Write a text message for this intent: {my_intent}

Examples of how I text this person:
{style_examples}

Context:
{context}

Write ONLY the message I would send, matching my style. Keep it short and natural.

Message:"""

        try:
            payload = {
                "model": self.base_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_tokens": 50  # Force short responses
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                cleaned_message = self._clean_generated_message(generated_text)
                
                return {
                    'message': cleaned_message,
                    'intent': my_intent,
                    'model': self.base_model,
                    'is_fine_tuned': False,
                    'context_messages': len(self.messages)
                }
            else:
                return {'error': f"Base model failed: {response.status_code}"}
                
        except Exception as e:
            return {'error': f"Base model error: {e}"}
    
    def generate_message(self, my_intent: str) -> dict:
        """Generate how I would text my intent to this person"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            gen_task = progress.add_task("✍️ Generating your message...", total=100)
            
            # Check model availability
            progress.update(gen_task, advance=20, description="🔍 Checking model availability...")
            has_fine_tuned = self._check_fine_tuned_model()
            model_to_use = self.model_name if has_fine_tuned else self.base_model
            
            console.print(f"🤖 Using model: {model_to_use}")
            
            # Prepare prompt
            progress.update(gen_task, advance=30, description="📝 Preparing context...")
            context = self._get_context()
            
            if has_fine_tuned:
                # Use fine-tuned model with clear instruction formatting
                prompt = f"""You are helping me write a text message. Based on our conversation history, write ONLY how I would naturally text them about: {my_intent}

Context (for reference only):
{context}

Intent: {my_intent}

Instructions:
- Write ONLY the text message I would send
- Match my natural texting style
- Keep it conversational and authentic
- Do NOT include conversation history or metadata
- Just return the message text, nothing else

My message:"""
            else:
                # Use more detailed context-based prompting
                your_messages = [m['text'] for m in self.messages if m['is_from_me']]
                style_examples = "\n".join([f"- {msg}" for msg in your_messages[-10:]])  # Last 10 messages as examples
                
                prompt = f"""Based on this conversation history, write how I would naturally text this person about: "{my_intent}"

Recent conversation context:
{context}

Examples of how I text this person:
{style_examples}

Write a message that:
1. Matches my natural texting style and tone with this specific person
2. Expresses the intent: {my_intent}
3. Fits the relationship dynamic shown in our conversation history
4. Uses my typical vocabulary, punctuation, and message length

Just return the message text, nothing else."""
            
            # Generate message
            progress.update(gen_task, advance=30, description="🧠 Generating your message...")
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
                
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
                
                progress.update(gen_task, advance=20, description="✅ Message generated!")
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '').strip()
                    
                    # Clean up the output
                    cleaned_message = self._clean_generated_message(generated_text)
                    
                    # If fine-tuned model produced garbage, try base model
                    if (has_fine_tuned and 
                        (len(cleaned_message) > 150 or 
                         cleaned_message.startswith("Error:") or
                         "conversation history" in cleaned_message.lower())):
                        
                        console.print("⚠️  Fine-tuned model output unclear, trying base model...")
                        return self._generate_with_base_model(my_intent)
                    
                    return {
                        'message': cleaned_message,
                        'intent': my_intent,
                        'model': model_to_use,
                        'is_fine_tuned': has_fine_tuned,
                        'context_messages': len(self.messages)
                    }
                else:
                    progress.update(gen_task, description="❌ Ollama request failed!")
                    return {'error': f"Ollama error: {response.status_code}"}
                    
            except Exception as e:
                progress.update(gen_task, description="❌ Generation failed!")
                return {'error': f"Connection error: {e}"}

    def generate_response(self, their_message: str) -> dict:
        """Generate response to their message"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            gen_task = progress.add_task("🤖 Generating response...", total=100)
            
            # Check model availability
            progress.update(gen_task, advance=20, description="🔍 Checking model availability...")
            has_fine_tuned = self._check_fine_tuned_model()
            model_to_use = self.model_name if has_fine_tuned else self.base_model
            
            console.print(f"🤖 Using model: {model_to_use}")
            
            # Prepare prompt
            progress.update(gen_task, advance=30, description="📝 Preparing prompt...")
            if has_fine_tuned:
                # Use fine-tuned model directly
                prompt = their_message
            else:
                # Use context-based prompting
                context = self._get_context()
                prompt = f"""Based on this conversation history, respond to their message in my natural texting style:

{context}

Their message: {their_message}

Respond as I would naturally text this person. Be authentic to my style shown above."""
            
            # Generate response
            progress.update(gen_task, advance=30, description="🧠 Generating response...")
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
                
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
                
                progress.update(gen_task, advance=20, description="✅ Response generated!")
                
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
                    progress.update(gen_task, description="❌ Ollama request failed!")
                    return {'error': f"Ollama error: {response.status_code}"}
                    
            except Exception as e:
                progress.update(gen_task, description="❌ Generation failed!")
                return {'error': f"Connection error: {e}"}
    
    def ask_question(self, question: str) -> dict:
        """Ask a question about the conversation history"""
        if not self.rag:
            return {'error': 'RAG system not available'}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            
            search_task = progress.add_task("🔍 Analyzing conversation history...", total=100)
            
            progress.update(search_task, advance=50, description="🔍 Searching relevant conversations...")
            result = self.rag.answer_question(question)
            
            progress.update(search_task, completed=100, description="✅ Analysis complete!")
            
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
        elif cmd == '/write' or cmd == '/message':
            if len(parts) < 2:
                console.print("❌ Usage: /write <what you want to say>")
                console.print("Example: /write ask them how their day is going")
                return
            intent = parts[1]
            result = self.generate_message(intent)
            self._display_generated_message(result)
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
        help_text = f"[bold cyan]TextTwin Commands[/bold cyan]\n\n"
        help_text += f"[bold]Message Generation:[/bold]\n"
        help_text += f"• /write <intent> → Generate how YOU would text them about something\n"
        help_text += f"  Example: /write ask them how their day is going\n"
        help_text += f"  Example: /write suggest we meet up for coffee\n"
        help_text += f"• Just type a message → Get a response to what they texted you\n\n"
        
        if RAG_AVAILABLE:
            help_text += f"[bold]Conversation Memory:[/bold]\n"
            help_text += f"• /ask <question> → Ask about your conversation history\n"
            help_text += f"• /search <query> → Search for specific conversations\n"
            help_text += f"• /stats → Show conversation statistics\n\n"
        else:
            help_text += f"[bold dim]Conversation Memory:[/bold dim] [dim]Unavailable (missing dependencies)[/dim]\n\n"
        
        help_text += f"[bold]General:[/bold]\n"
        help_text += f"• /help → Show this help\n"
        help_text += f"• quit → Exit"
        
        console.print(Panel(help_text, style="blue"))
    
    def _display_generated_message(self, result: dict):
        """Display generated message result"""
        if 'error' in result:
            console.print(f"❌ {result['error']}")
            return
        
        # Show the generated message prominently
        console.print(f"\n✍️ [bold green]Your message:[/bold green]")
        console.print(f"[bold cyan]\"{result['message']}\"[/bold cyan]")
        
        # Show intent and model info
        console.print(f"\n💭 Intent: {result['intent']}")
        console.print(f"🤖 Model: {result['model']} | Context: {result['context_messages']} messages")
        
        if result.get('is_fine_tuned'):
            console.print("✨ Using your personalized fine-tuned model")
        else:
            console.print("📚 Using context-based style matching")
    
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
        
        if not RAG_AVAILABLE:
            rag_status = "❌ Dependencies missing"
        elif self.rag:
            rag_status = "✅ Available"
        else:
            rag_status = "❌ Not initialized"
        
        intro_text = f"🤖 [bold cyan]TextTwin Interactive Mode[/bold cyan]\n\n"
        intro_text += f"Phone: {self.phone_number}\n"
        intro_text += f"Messages in history: {len(self.messages)}\n"
        intro_text += f"Fine-tuned model: {'✅ Available' if self._check_fine_tuned_model() else '❌ Not found'}\n"
        intro_text += f"Conversation Memory: {rag_status}\n\n"
        intro_text += f"[bold]Message Generation:[/bold]\n"
        intro_text += f"• /write <intent> → Generate how YOU would text them\n"
        intro_text += f"  Example: /write ask how their day went\n"
        intro_text += f"• Type their message → Get your response style\n\n"
        
        if RAG_AVAILABLE and self.rag:
            intro_text += f"[bold]Memory Commands:[/bold]\n"
            intro_text += f"• /ask <question> → Ask about conversation history\n"
            intro_text += f"• /search <query> → Search conversations\n"
            intro_text += f"• /stats → Show statistics\n\n"
        
        intro_text += f"[bold]Other:[/bold] /help → Help | quit → Exit"
        
        console.print(Panel(intro_text, style="cyan"))
        
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