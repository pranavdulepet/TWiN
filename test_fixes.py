#!/usr/bin/env python3
"""
Quick test to verify the fixes work
"""

import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from simple_rag import SimpleRAG
from rich.console import Console

console = Console()

def test_answer_question_fix():
    """Test that answer_question method works without crashing"""
    console.print("🧪 Testing answer_question fix...")
    
    try:
        # Create RAG instance (will work even without data)
        rag = SimpleRAG("1234567890")
        
        # Test the query that was causing issues
        result = rag.answer_question("who is this person")
        
        # Check that result is properly formed
        if result is None:
            console.print("❌ Result is still None")
            return False
        
        if not isinstance(result, dict):
            console.print(f"❌ Result is not a dict: {type(result)}")
            return False
        
        if 'answer' not in result:
            console.print("❌ Result missing 'answer' key")
            return False
        
        console.print(f"✅ Answer: {result['answer']}")
        console.print(f"✅ Confidence: {result.get('confidence', 'N/A')}")
        console.print(f"✅ Sources: {len(result.get('sources', []))}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error in answer_question: {e}")
        return False

if __name__ == "__main__":
    console.print("🔧 Testing RAG System Fixes")
    console.print("=" * 30)
    
    success = test_answer_question_fix()
    
    if success:
        console.print("\n✅ All fixes working correctly!")
        console.print("🎯 The system is ready for the user to try again.")
    else:
        console.print("\n❌ Some issues remain.")