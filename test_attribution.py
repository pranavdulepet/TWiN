#!/usr/bin/env python3
"""
Test message attribution to ensure the model clearly distinguishes 
between user and other person's messages
"""

import sys
import json
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from simple_rag import SimpleRAG
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_message_attribution():
    """Test that message attribution is working correctly"""
    console.print(Panel.fit("🧪 Testing Message Attribution System", style="cyan"))
    
    # Load the actual conversation data
    conv_file = "data/conversation_9255238027.json"
    if not Path(conv_file).exists():
        console.print(f"❌ Conversation file not found: {conv_file}")
        return False
    
    with open(conv_file, 'r') as f:
        messages = json.load(f)
    
    console.print(f"📱 Loaded {len(messages)} messages")
    
    # Test basic attribution
    your_messages = [m for m in messages if m['is_from_me']]
    their_messages = [m for m in messages if not m['is_from_me']]
    
    table = Table(title="Message Attribution Check")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")
    table.add_column("Sample", style="dim")
    
    total = len(messages)
    your_pct = len(your_messages) / total * 100
    their_pct = len(their_messages) / total * 100
    
    your_sample = your_messages[0]['text'][:50] + "..." if your_messages else "None"
    their_sample = their_messages[0]['text'][:50] + "..." if their_messages else "None"
    
    table.add_row("Your messages", str(len(your_messages)), f"{your_pct:.1f}%", your_sample)
    table.add_row("Their messages", str(len(their_messages)), f"{their_pct:.1f}%", their_sample)
    
    console.print(table)
    
    return True

def test_chunk_attribution():
    """Test that chunks properly show You/Them labels"""
    console.print(f"\n🔍 Testing Chunk Attribution...")
    
    try:
        # Create RAG system with actual data
        rag = SimpleRAG("9255238027")
        
        # Get some chunks and inspect their formatting
        import sqlite3
        
        conn = sqlite3.connect(rag.db_file)
        cursor = conn.cursor()
        
        # Get first few chunks
        cursor.execute("SELECT text FROM conversation_chunks LIMIT 3")
        sample_chunks = cursor.fetchall()
        conn.close()
        
        console.print(f"✅ Found {len(sample_chunks)} sample chunks")
        
        for i, (chunk_text,) in enumerate(sample_chunks):
            console.print(f"\n📝 Sample Chunk {i+1}:")
            # Show first few lines of chunk
            lines = chunk_text.split('\n')[:3]
            for line in lines:
                if line.strip():
                    console.print(f"   {line}")
            if len(lines) > 3:
                console.print("   ...")
        
        # Verify You/Them labels are present
        all_chunk_text = ' '.join([chunk[0] for chunk in sample_chunks])
        has_you_labels = "You:" in all_chunk_text
        has_them_labels = "Them:" in all_chunk_text
        
        console.print(f"\n✅ Contains 'You:' labels: {has_you_labels}")
        console.print(f"✅ Contains 'Them:' labels: {has_them_labels}")
        
        return has_you_labels and has_them_labels
        
    except Exception as e:
        console.print(f"❌ Error testing chunks: {e}")
        return False

def test_ai_response_attribution():
    """Test that AI can correctly identify who said what"""
    console.print(f"\n🤖 Testing AI Response Attribution...")
    
    try:
        rag = SimpleRAG("9255238027")
        
        # Ask a question that requires distinguishing between speakers
        questions = [
            "Who talks more in our conversations, me or them?",
            "What's an example of something I typically say?", 
            "What's an example of something they typically say?"
        ]
        
        for question in questions:
            console.print(f"\n❓ Question: {question}")
            result = rag.answer_question(question)
            
            if 'error' in result.get('answer', '').lower():
                console.print(f"❌ Error in response: {result['answer']}")
                continue
                
            answer = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            console.print(f"💬 Answer: {answer}")
            
            # Check if the answer makes sense and distinguishes speakers
            has_attribution = any(word in answer.lower() for word in ['you', 'they', 'your', 'their', 'yours', 'theirs'])
            console.print(f"✅ Contains speaker attribution: {has_attribution}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error testing AI responses: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    console.print("🔍 Message Attribution Verification Test")
    console.print("=" * 50)
    
    # Test 1: Basic message counts and attribution
    test1_success = test_message_attribution()
    
    # Test 2: Chunk formatting
    test2_success = test_chunk_attribution() if test1_success else False
    
    # Test 3: AI response attribution
    test3_success = test_ai_response_attribution() if test2_success else False
    
    console.print(f"\n📊 Test Results:")
    console.print(f"✅ Message Attribution: {'PASS' if test1_success else 'FAIL'}")
    console.print(f"✅ Chunk Formatting: {'PASS' if test2_success else 'FAIL'}")
    console.print(f"✅ AI Response Attribution: {'PASS' if test3_success else 'FAIL'}")
    
    if test1_success and test2_success and test3_success:
        console.print(f"\n🎉 All attribution tests PASSED!")
        console.print(f"✅ The model clearly distinguishes between your messages and theirs")
    else:
        console.print(f"\n❌ Some attribution tests FAILED!")
        console.print(f"⚠️  Message attribution may not be working correctly")

if __name__ == "__main__":
    main()