#!/usr/bin/env python3
"""
TextTwin Engine - Personal Writing Style Cloner
===============================================

The main engine that combines message analysis with local LLM to create
a personalized texting AI that mimics your writing style.
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from message_analyzer import MessageAnalyzer, MessageStyle
from safe_imessage_reader import SafeIMessageReader

console = Console()

class TextTwinEngine:
    """Main engine for creating personalized text responses."""
    
    def __init__(self):
        self.analyzer = MessageAnalyzer()
        self.style_profile: Optional[MessageStyle] = None
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"
        
    def load_style_from_file(self, filepath: str) -> None:
        """Load a previously saved style profile."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct MessageStyle object
        self.style_profile = MessageStyle(
            avg_message_length=data['avg_message_length'],
            avg_words_per_message=data['avg_words_per_message'],
            avg_sentences_per_message=data['avg_sentences_per_message'],
            punctuation_habits=data['punctuation_habits'],
            capitalization_ratio=data['capitalization_ratio'],
            emoji_usage_rate=data['emoji_usage_rate'],
            common_phrases=data['common_phrases'],
            response_time_patterns=data.get('response_time_patterns', {}),
            conversation_starters=data['conversation_starters'],
            unique_vocabulary=data['unique_vocabulary'],
            readability_score=data['readability_score'],
            sentiment_indicators=data['sentiment_indicators'],
            time_of_day_patterns=data['time_of_day_patterns']
        )
        
        console.print(f"✓ Style profile loaded from {filepath}")
    
    def analyze_messages_from_csv(self, csv_path: str) -> None:
        """Analyze messages from a CSV file."""
        self.analyzer.load_from_csv(csv_path)
        self.style_profile = self.analyzer.analyze_style()
        console.print("✓ Message analysis complete")
    
    def analyze_sample_messages(self) -> None:
        """Analyze sample messages for demo purposes."""
        self.analyzer.load_from_sample_data()
        self.style_profile = self.analyzer.analyze_style()
        console.print("✓ Sample message analysis complete")
    
    def analyze_imessages(self) -> None:
        """Analyze real iMessage data safely."""
        try:
            with SafeIMessageReader() as reader:
                messages_df = reader.get_user_messages_only(limit=1000)
                
                # Convert to expected format
                formatted_messages = []
                for _, row in messages_df.iterrows():
                    timestamp = datetime.fromtimestamp(row['date'] / 1000000000 + 978307200)  # Apple timestamp conversion
                    formatted_messages.append({
                        'text': row['text'],
                        'timestamp': timestamp.isoformat(),
                        'is_from_me': bool(row['is_from_me'])
                    })
                
                # Create DataFrame and analyze
                self.analyzer.messages_df = pd.DataFrame(formatted_messages)
                self.analyzer.messages_df['timestamp'] = pd.to_datetime(self.analyzer.messages_df['timestamp'])
                
                self.style_profile = self.analyzer.analyze_style()
                console.print("✓ iMessage analysis complete")
                
        except Exception as e:
            console.print(f"[red]Could not analyze iMessages: {e}[/red]")
            console.print("[yellow]Falling back to sample data for demo[/yellow]")
            self.analyze_sample_messages()
    
    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model_name in model_names:
                    console.print(f"✓ Connected to Ollama - {self.model_name} available")
                    return True
                else:
                    console.print(f"[red]Model {self.model_name} not found. Available models: {model_names}[/red]")
                    return False
            else:
                console.print("[red]Ollama server not responding[/red]")
                return False
        except Exception as e:
            console.print(f"[red]Cannot connect to Ollama: {e}[/red]")
            return False
    
    def create_style_prompt(self, context: str = "") -> str:
        """Create a detailed prompt that captures the user's texting style."""
        if not self.style_profile:
            raise ValueError("No style profile available. Analyze messages first.")
        
        # Build a comprehensive style description
        style_description = f"""You are mimicking a specific person's texting style. Here are their characteristics:

WRITING PATTERNS:
- Average message length: {self.style_profile.avg_message_length:.1f} characters
- Average words per message: {self.style_profile.avg_words_per_message:.1f}
- Capitalization style: {self.style_profile.capitalization_ratio:.1%} of letters are capitals
- Emoji usage: {self.style_profile.emoji_usage_rate:.2f} emojis per message

PUNCTUATION HABITS:"""
        
        for punct, rate in self.style_profile.punctuation_habits.items():
            style_description += f"\n- Uses '{punct}' about {rate:.2f} times per message"
        
        style_description += f"\n\nCOMMON PHRASES:"
        for phrase, count in self.style_profile.common_phrases[:8]:
            style_description += f"\n- Often says: '{phrase}'"
        
        style_description += f"\n\nCONVERSATION STARTERS:"
        for starter in self.style_profile.conversation_starters[:5]:
            style_description += f"\n- '{starter}'"
        
        style_description += f"\n\nPERSONALITY INDICATORS:"
        style_description += f"\n- Readability level: {self.style_profile.readability_score:.1f}/100 (higher = simpler)"
        style_description += f"\n- Uses {self.style_profile.unique_vocabulary} unique words"
        
        if self.style_profile.sentiment_indicators:
            total_sentiment = sum(self.style_profile.sentiment_indicators.values())
            if total_sentiment > 0:
                for sentiment, count in self.style_profile.sentiment_indicators.items():
                    percentage = (count / total_sentiment) * 100
                    style_description += f"\n- {sentiment.title()} expressions: {percentage:.1f}%"
        
        prompt = f"""{style_description}

INSTRUCTIONS:
1. Respond EXACTLY as this person would text
2. Match their message length, punctuation, and emoji usage
3. Use their common phrases naturally
4. Maintain their personality and tone
5. Keep responses casual and authentic
6. Don't explain that you're mimicking - just BE them

{f"CONTEXT: {context}" if context else ""}

Respond as this person would:"""
        
        return prompt
    
    def generate_response(self, input_message: str, context: str = "") -> str:
        """Generate a response in the user's style using the local LLM."""
        if not self.style_profile:
            raise ValueError("No style profile available. Analyze messages first.")
        
        style_prompt = self.create_style_prompt(context)
        full_prompt = f"{style_prompt}\n\nMessage to respond to: \"{input_message}\"\n\nResponse:"
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating response...", total=None)
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "max_tokens": 150
                        }
                    },
                    timeout=30
                )
                
                progress.update(task, completed=100)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Clean up the response
                if generated_text:
                    # Remove any quotes or formatting artifacts
                    generated_text = generated_text.strip('"\'')
                    return generated_text
                else:
                    return "sorry, couldn't think of anything to say"
            else:
                console.print(f"[red]API Error: {response.status_code}[/red]")
                return "hmm something went wrong"
                
        except Exception as e:
            console.print(f"[red]Generation error: {e}[/red]")
            return "oops my brain isn't working rn"
    
    def interactive_chat(self) -> None:
        """Start an interactive chat session."""
        if not self.style_profile:
            console.print("[red]No style profile loaded. Please analyze messages first.[/red]")
            return
        
        if not self.test_ollama_connection():
            console.print("[red]Cannot connect to Ollama. Please ensure it's running.[/red]")
            return
        
        console.print(Panel(
            "🤖 [bold cyan]TextTwin Interactive Chat[/bold cyan]\n\n" +
            "You're now chatting with an AI trained on your texting style!\n" +
            "Type messages and see how you would typically respond.\n\n" +
            "[dim]Type 'quit' to exit[/dim]",
            style="blue"
        ))
        
        conversation_history = []
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]Message[/bold green]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[dim]Thanks for chatting! 👋[/dim]")
                    break
                
                # Build context from recent conversation
                context = ""
                if conversation_history:
                    recent_context = conversation_history[-3:]  # Last 3 exchanges
                    context = "Recent conversation: " + " | ".join([
                        f"Them: {msg['input']} You: {msg['response']}" 
                        for msg in recent_context
                    ])
                
                # Generate response
                response = self.generate_response(user_input, context)
                
                # Display response
                console.print(f"[bold blue]You would respond:[/bold blue] {response}")
                
                # Add to conversation history
                conversation_history.append({
                    'input': user_input,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                console.print("\n[dim]Chat ended.[/dim]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def batch_responses(self, input_file: str, output_file: str) -> None:
        """Generate responses for a batch of messages from a file."""
        if not self.style_profile:
            console.print("[red]No style profile loaded. Please analyze messages first.[/red]")
            return
        
        with open(input_file, 'r') as f:
            messages = [line.strip() for line in f if line.strip()]
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Generating responses...", total=len(messages))
            
            for i, message in enumerate(messages):
                response = self.generate_response(message)
                results.append({
                    'input': message,
                    'response': response,
                    'generated_at': datetime.now().isoformat()
                })
                progress.update(task, advance=1)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"✓ Generated {len(results)} responses saved to {output_file}")
    
    def display_style_summary(self) -> None:
        """Display a summary of the analyzed style."""
        if not self.style_profile:
            console.print("[red]No style profile available.[/red]")
            return
        
        self.analyzer.display_style_report()


def main():
    """Main CLI interface for TextTwin."""
    console.print(Panel(
        "🎯 [bold magenta]TextTwin - Personal Writing Style Cloner[/bold magenta]\n\n" +
        "Analyze your messages and create an AI that texts just like you!",
        style="magenta"
    ))
    
    engine = TextTwinEngine()
    
    while True:
        console.print("\n[bold cyan]What would you like to do?[/bold cyan]")
        
        table = Table(style="cyan")
        table.add_column("Option", style="bold")
        table.add_column("Description")
        
        table.add_row("1", "analyze sample messages (demo)")
        table.add_row("2", "analyze real iMessages")
        table.add_row("3", "load saved style profile")
        table.add_row("4", "view style summary")
        table.add_row("5", "start interactive chat")
        table.add_row("6", "test Ollama connection")
        table.add_row("q", "quit")
        
        console.print(table)
        
        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6", "q"])
        
        try:
            if choice == "1":
                engine.analyze_sample_messages()
                engine.analyzer.save_style_profile('my_texting_style.json')
            
            elif choice == "2":
                engine.analyze_imessages()
                engine.analyzer.save_style_profile('my_texting_style.json')
            
            elif choice == "3":
                filepath = Prompt.ask("Enter path to style profile JSON file", default="my_texting_style.json")
                engine.load_style_from_file(filepath)
            
            elif choice == "4":
                engine.display_style_summary()
            
            elif choice == "5":
                engine.interactive_chat()
            
            elif choice == "6":
                engine.test_ollama_connection()
            
            elif choice == "q":
                console.print("[dim]Goodbye! 👋[/dim]")
                break
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()