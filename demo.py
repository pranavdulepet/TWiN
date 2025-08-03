#!/usr/bin/env python3
"""
TextTwin Demo Script
===================

Demonstrates the TextTwin system capabilities without interactive prompts.
"""

from texttwin_engine import TextTwinEngine
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_demo():
    """Run a complete demonstration of TextTwin."""
    
    console.print(Panel(
        "🎯 [bold magenta]TextTwin Demo - Personal Writing Style Cloner[/bold magenta]\n\n" +
        "This demo will:\n" +
        "1. Analyze sample messages to learn texting style\n" +
        "2. Test connection to local LLM (Ollama)\n" +
        "3. Generate responses in your style\n" +
        "4. Show how the AI mimics your texting patterns",
        style="magenta"
    ))
    
    # Initialize the engine
    engine = TextTwinEngine()
    
    # Step 1: Analyze sample messages
    console.print("\n[bold cyan]Step 1: Analyzing sample messages...[/bold cyan]")
    engine.analyze_sample_messages()
    
    # Step 2: Display style analysis
    console.print("\n[bold cyan]Step 2: Your texting style profile[/bold cyan]")
    engine.display_style_summary()
    
    # Step 3: Test Ollama connection
    console.print("\n[bold cyan]Step 3: Testing AI connection...[/bold cyan]")
    if not engine.test_ollama_connection():
        console.print("[red]❌ Cannot connect to Ollama. Make sure it's running with 'brew services start ollama'[/red]")
        return
    
    # Step 4: Generate sample responses
    console.print("\n[bold cyan]Step 4: Generating responses in your style...[/bold cyan]")
    
    test_messages = [
        "hey what's up?",
        "want to grab dinner tonight?",
        "did you see that crazy thing on the news?",
        "I'm so tired from work today",
        "what are you doing this weekend?"
    ]
    
    for msg in test_messages:
        console.print(f"\n[bold green]📱 Someone texts you:[/bold green] \"{msg}\"")
        response = engine.generate_response(msg)
        console.print(f"[bold blue]🤖 You would respond:[/bold blue] \"{response}\"")
    
    # Step 5: Show style comparison
    console.print("\n[bold cyan]Step 5: Style Analysis Summary[/bold cyan]")
    
    style = engine.style_profile
    console.print(Panel(
        f"📊 [bold]Your Texting Personality:[/bold]\n\n" +
        f"• Average message length: {style.avg_message_length:.1f} characters\n" +
        f"• Words per message: {style.avg_words_per_message:.1f}\n" +
        f"• Emoji usage: {style.emoji_usage_rate:.2f} per message\n" +
        f"• Readability score: {style.readability_score:.1f}/100\n" +
        f"• Unique vocabulary: {style.unique_vocabulary} words\n\n" +
        f"The AI learned these patterns and will mimic:\n" +
        f"• Your punctuation habits\n" +
        f"• Your common phrases\n" +
        f"• Your conversation style\n" +
        f"• Your emoji preferences",
        title="Analysis Complete",
        style="green"
    ))
    
    console.print(Panel(
        "🎉 [bold green]Demo Complete![/bold green]\n\n" +
        "TextTwin has successfully:\n" +
        "✅ Analyzed your texting patterns\n" +
        "✅ Connected to local AI (Ollama)\n" +
        "✅ Generated responses in your style\n" +
        "✅ Maintained complete privacy (all local)\n\n" +
        "[dim]To use with real iMessages, ensure you have proper permissions and run the full application.[/dim]",
        style="green"
    ))


if __name__ == "__main__":
    run_demo()