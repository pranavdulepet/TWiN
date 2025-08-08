#!/usr/bin/env python3
"""
Test script to verify model update works correctly
"""

import sys
import json
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from texttwin import TextTwin

def test_model_selection():
    """Test that the model selection logic works"""
    print("🧪 Testing model selection logic...")
    
    # Create a dummy TextTwin instance
    twin = TextTwin("1234567890")
    
    # Test model selection
    best_model = twin._get_best_available_model()
    print(f"✅ Best available model: {best_model}")
    
    # Test model health check
    health_check = twin._test_model_health(best_model)
    print(f"✅ Model health check: {'PASSED' if health_check else 'FAILED'}")
    
    return best_model, health_check

def test_message_generation():
    """Test basic message generation"""
    print("\n🧪 Testing message generation...")
    
    try:
        twin = TextTwin("1234567890")
        result = twin._generate_with_base_model("say hello")
        
        if 'error' in result:
            print(f"❌ Message generation failed: {result['error']}")
            return False
        else:
            print(f"✅ Generated message: '{result['message']}'")
            print(f"✅ Used model: {result['model']}")
            return True
            
    except Exception as e:
        print(f"❌ Exception during message generation: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Testing TextTwin Model Update")
    print("=" * 40)
    
    # Test model selection
    model, health = test_model_selection()
    
    if not health:
        print("❌ Model health check failed - something is wrong with Ollama setup")
        sys.exit(1)
    
    # Test message generation
    success = test_message_generation()
    
    if success:
        print("\n✅ All tests passed! The system is ready to use.")
        print(f"🎯 Default model priority: gpt-oss:20b → llama3.2:3b")
        print(f"🎯 Currently using: {model}")
    else:
        print("\n❌ Some tests failed. Check your setup.")
        sys.exit(1)