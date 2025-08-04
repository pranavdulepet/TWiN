#!/usr/bin/env python3
"""
Find the best contact for fine-tuning (one with actual text conversations)
"""

import sqlite3
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def find_best_contacts_for_training():
    """Find contacts with the most actual text conversation pairs."""
    
    console.print("🔍 [bold cyan]Finding contacts with best text conversations for fine-tuning...[/bold cyan]")
    
    imessage_db = Path.home() / "Library" / "Messages" / "chat.db"
    
    try:
        with sqlite3.connect(f"file:{imessage_db}?mode=ro", uri=True) as conn:
            # Get contacts with most bidirectional text conversations
            query = """
            SELECT 
                h.id as phone_number,
                COUNT(*) as total_messages,
                SUM(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as your_messages,
                SUM(CASE WHEN m.is_from_me = 0 THEN 1 ELSE 0 END) as their_messages
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL 
            AND m.text != ''
            AND length(m.text) > 3
            AND m.text NOT LIKE 'Loved %'
            AND m.text NOT LIKE 'Liked %' 
            AND m.text NOT LIKE 'Laughed at %'
            AND m.text NOT LIKE 'Emphasized %'
            AND m.text NOT LIKE 'Gaf een hartje aan %'
            AND m.text NOT LIKE 'Lachte om %'
            AND m.text NOT LIKE '%￼%'
            GROUP BY h.id
            HAVING your_messages > 5 AND their_messages > 5
            ORDER BY (your_messages + their_messages) DESC
            LIMIT 15
            """
            
            cursor = conn.execute(query)
            results = cursor.fetchall()
            
            if not results:
                console.print("❌ No contacts with sufficient text conversations found")
                return
            
            console.print(f"✅ Found {len(results)} contacts with good text conversations")
            
            # Create table
            table = Table(title="📱 Best Contacts for Fine-Tuning")
            table.add_column("Phone Number", style="cyan")
            table.add_column("Total Msgs", justify="right")
            table.add_column("Your Msgs", justify="right", style="green")
            table.add_column("Their Msgs", justify="right", style="blue")
            table.add_column("Ratio", justify="right")
            
            for phone, total, yours, theirs in results:
                ratio = f"{yours/theirs:.1f}" if theirs > 0 else "∞"
                table.add_row(
                    phone, 
                    str(total), 
                    str(yours), 
                    str(theirs), 
                    ratio
                )
            
            console.print(table)
            
            # Show sample conversation from top contact
            if results:
                top_contact = results[0][0]
                console.print(f"\n💬 Sample conversation with {top_contact}:")
                
                sample_query = """
                SELECT m.text, m.is_from_me, m.date
                FROM message m
                JOIN handle h ON m.handle_id = h.ROWID
                WHERE h.id = ?
                AND m.text IS NOT NULL 
                AND m.text != ''
                AND length(m.text) > 3
                AND m.text NOT LIKE 'Loved %'
                AND m.text NOT LIKE 'Liked %'
                AND m.text NOT LIKE '%￼%'
                ORDER BY m.date DESC
                LIMIT 10
                """
                
                cursor = conn.execute(sample_query, (top_contact,))
                sample_msgs = cursor.fetchall()
                
                for text, is_from_me, date in sample_msgs:
                    sender = "You" if is_from_me else "Them"
                    display_text = text[:80] + "..." if len(text) > 80 else text
                    console.print(f"   {sender}: {display_text}")
                
                console.print(f"\n🎯 [bold green]Recommendation: Use {top_contact} for fine-tuning![/bold green]")
                console.print(f"This contact has {results[0][2]} of your text messages and {results[0][3]} of theirs.")
                
    except Exception as e:
        console.print(f"❌ Database error: {e}")
        console.print("💡 Need Full Disk Access to analyze your contacts")

def analyze_specific_contact(phone_number: str):
    """Analyze a specific contact in detail."""
    
    console.print(f"\n🔍 [bold]Detailed analysis of {phone_number}:[/bold]")
    
    imessage_db = Path.home() / "Library" / "Messages" / "chat.db"
    
    try:
        with sqlite3.connect(f"file:{imessage_db}?mode=ro", uri=True) as conn:
            # Get all messages for this contact
            normalized = re.sub(r'[^\d+]', '', phone_number)
            if not normalized.startswith('+'):
                if len(normalized) == 10:
                    normalized = f"+1{normalized}"
                elif len(normalized) == 11 and normalized.startswith('1'):
                    normalized = f"+{normalized}"
            
            query = """
            SELECT m.text, m.is_from_me
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            WHERE h.id LIKE ?
            AND m.text IS NOT NULL 
            AND m.text != ''
            ORDER BY m.date ASC
            """
            
            cursor = conn.execute(query, (f"%{normalized.replace('+1', '')}%",))
            messages = cursor.fetchall()
            
            if not messages:
                console.print(f"❌ No messages found for {phone_number}")
                return
            
            console.print(f"📊 Found {len(messages)} total messages")
            
            # Categorize messages
            your_text = []
            your_reactions = []
            their_text = []
            their_reactions = []
            
            reaction_keywords = ['Loved', 'Liked', 'Laughed at', 'Emphasized', 'Gaf een hartje', 'Lachte om', '￼']
            
            for text, is_from_me in messages:
                is_reaction = any(keyword in text for keyword in reaction_keywords)
                
                if is_from_me:
                    if is_reaction:
                        your_reactions.append(text)
                    else:
                        your_text.append(text)
                else:
                    if is_reaction:
                        their_reactions.append(text)
                    else:
                        their_text.append(text)
            
            console.print(f"📤 Your messages:")
            console.print(f"   Text messages: {len(your_text)}")
            console.print(f"   Reactions: {len(your_reactions)}")
            
            console.print(f"📥 Their messages:")
            console.print(f"   Text messages: {len(their_text)}")
            console.print(f"   Reactions: {len(their_reactions)}")
            
            if len(your_text) < 5:
                console.print(f"\n⚠️ [yellow]This contact has very few text messages from you ({len(your_text)})[/yellow]")
                console.print("💡 You primarily communicate through reactions/images with this person")
                console.print("🎯 For better fine-tuning, use a contact where you exchange more text messages")
            else:
                console.print(f"\n✅ [green]This contact has {len(your_text)} text messages from you - good for training![/green]")
                
                # Show sample text messages
                console.print(f"\n💬 Sample of your text messages:")
                for i, text in enumerate(your_text[:5], 1):
                    display = text[:60] + "..." if len(text) > 60 else text
                    console.print(f"   {i}. \"{display}\"")
            
    except Exception as e:
        console.print(f"❌ Database error: {e}")

if __name__ == "__main__":
    find_best_contacts_for_training()
    
    # Also analyze the specific contact you tried
    print()
    analyze_specific_contact("XXXXXXXXXX")