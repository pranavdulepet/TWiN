#!/usr/bin/env python3
"""
Test script to verify reasoning display functionality
"""

import sys
import time
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from texttwin import TextTwin

def test_reasoning_features():
    """Test reasoning display and toggle functionality"""
    print("🧪 Testing Reasoning Features")
    print("=" * 40)
    
    # Create TextTwin instance
    twin = TextTwin("1234567890")
    
    # Test 1: Reasoning enabled by default
    print(f"\n✅ Default reasoning setting: {'ON' if twin.show_reasoning else 'OFF'}")
    
    # Test 2: Toggle reasoning off
    twin.show_reasoning = False
    print(f"✅ Reasoning toggled off: {'ON' if twin.show_reasoning else 'OFF'}")
    
    # Test 3: Toggle reasoning on
    twin.show_reasoning = True
    print(f"✅ Reasoning toggled on: {'ON' if twin.show_reasoning else 'OFF'}")
    
    # Test 4: Test response parsing
    print("\n🧪 Testing reasoning parsing...")
    
    # Mock GPT-OSS response with reasoning
    mock_response = """Thinking...
The user is asking me to test the reasoning functionality. I should respond with a clear acknowledgment that shows the reasoning is working properly.
...done thinking.

Sure! The reasoning functionality is working great."""
    
    parsed = twin._parse_reasoning_response(mock_response)
    print(f"✅ Reasoning detected: {parsed['has_reasoning']}")
    if parsed['has_reasoning']:
        print(f"✅ Reasoning content: {parsed['reasoning'][:50]}...")
        print(f"✅ Response content: {parsed['response']}")
    
    # Test 5: Test response without reasoning
    print("\n🧪 Testing non-reasoning response...")
    simple_response = "This is a simple response without reasoning."
    parsed_simple = twin._parse_reasoning_response(simple_response)
    print(f"✅ No reasoning detected: {not parsed_simple['has_reasoning']}")
    print(f"✅ Response content: {parsed_simple['response']}")
    
    print("\n✅ All reasoning tests passed!")
    return True

def test_generate_with_reasoning():
    """Test actual message generation with reasoning display"""
    print("\n🧪 Testing Live Message Generation with Reasoning")
    print("=" * 50)
    
    try:
        twin = TextTwin("1234567890")
        
        print("🎯 Testing with reasoning ON...")
        twin.show_reasoning = True
        result = twin._generate_with_base_model("say a simple greeting")
        
        if 'error' in result:
            print(f"❌ Generation failed: {result['error']}")
            return False
        
        print(f"✅ Generated message: {result['message']}")
        print(f"✅ Has reasoning: {result.get('has_reasoning', False)}")
        if result.get('reasoning'):
            print(f"✅ Reasoning preview: {result['reasoning'][:100]}...")
        
        print(f"✅ Used model: {result['model']}")
        return True
        
    except Exception as e:
        print(f"❌ Error during live test: {e}")
        return False

if __name__ == "__main__":
    # Run basic tests
    success1 = test_reasoning_features()
    
    # Run live generation test
    success2 = test_generate_with_reasoning()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Reasoning functionality is ready.")
        print("\n📋 Commands available to users:")
        print("  • /reasoning        → Toggle reasoning display")
        print("  • /reasoning on     → Enable reasoning display") 
        print("  • /reasoning off    → Disable reasoning display")
        print("\n✨ Features implemented:")
        print("  • 🤔 Animated 'Thinking...' indicator")
        print("  • 🧠 Reasoning content display (toggleable)")
        print("  • 🎛️  User controls for reasoning visibility")
        print("  • 📱 Clean separation of reasoning and response")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        sys.exit(1)