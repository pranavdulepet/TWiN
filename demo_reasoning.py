#!/usr/bin/env python3
"""
Demo script showing reasoning functionality in action
"""

import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from texttwin import TextTwin
from rich.console import Console
from rich.panel import Panel

console = Console()

def demo_reasoning_features():
    """Demonstrate the reasoning features"""
    console.print(Panel.fit(
        "🧠 TextTwin Reasoning Demo\n"
        "This demo shows the new reasoning display features", 
        style="cyan"
    ))
    
    # Create TextTwin instance
    twin = TextTwin("1234567890")
    
    console.print("\n📋 [bold]Available Commands:[/bold]")
    console.print("• `/reasoning` - Toggle reasoning display on/off")
    console.print("• `/reasoning on` - Enable reasoning display")  
    console.print("• `/reasoning off` - Disable reasoning display")
    
    console.print(f"\n🎛️  [bold]Current Setting:[/bold] Reasoning display is {'ON' if twin.show_reasoning else 'OFF'}")
    
    console.print("\n✨ [bold]Key Features Implemented:[/bold]")
    console.print("1. 🤔 Animated 'Thinking...' indicator during model reasoning")
    console.print("2. 🧠 Display model's chain-of-thought process (when available)")
    console.print("3. 🎛️  User toggle to show/hide reasoning")
    console.print("4. 📱 Clean separation of reasoning from final response")
    console.print("5. ⚡ Works with GPT-OSS models' native reasoning capabilities")
    
    console.print("\n🔍 [bold]When You'll See Reasoning:[/bold]")
    console.print("• GPT-OSS-20B responses with 'Thinking...' sections")
    console.print("• Complex requests that trigger model reasoning")
    console.print("• Question answering with RAG system")
    
    console.print("\n🎯 [bold]User Experience:[/bold]")
    console.print("1. User sends a message")
    console.print("2. [cyan]🤔 Thinking...[/cyan] indicator appears (if reasoning enabled)")
    console.print("3. Model reasoning is displayed (if available and enabled)")
    console.print("4. Final response is shown cleanly")
    console.print("5. Toggle status is shown for reference")
    
    console.print(f"\n🚀 [bold green]System Ready![/bold green] Use TextTwin normally - reasoning will appear automatically when the model provides it.")

if __name__ == "__main__":
    demo_reasoning_features()