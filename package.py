"""
TextTwin Packaging Script
Creates a standalone distribution for the XXXXXXXXXX contact
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

def create_package():
    """Create standalone TextTwin package"""
    
    # Package directory
    package_dir = Path("TextTwin-XXXXXXXXXX-Standalone")
    package_dir.mkdir(exist_ok=True)
    
    print("📦 Creating TextTwin standalone package...")
    
    # 1. Copy core application files
    core_files = [
        "texttwin.py",
        "simple_rag.py", 
        "scripts/message_extractor.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in core_files:
        if Path(file).exists():
            dest = package_dir / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            print(f"✅ Copied {file}")
    
    # 2. Copy conversation data
    data_files = [
        "data/conversation_XXXXXXXXXX.json",
        "data/rag_XXXXXXXXXX.db",
        "data/rag_XXXXXXXXXX.faiss"
    ]
    
    data_dir = package_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    for file in data_files:
        if Path(file).exists():
            shutil.copy2(file, data_dir / Path(file).name)
            print(f"✅ Copied {file}")
    
    # 3. Copy fine-tuned adapter
    adapter_source = Path("adapters/texttwin-XXXXXXXXXX")
    adapter_dest = package_dir / "adapters/texttwin-XXXXXXXXXX"
    
    if adapter_source.exists():
        shutil.copytree(adapter_source, adapter_dest, dirs_exist_ok=True)
        print("✅ Copied fine-tuned adapter")
    
    # 4. Create Ollama setup script
    setup_script = package_dir / "setup.sh"
    setup_script.write_text("""#!/bin/bash
# TextTwin Setup Script

echo "🤖 Setting up TextTwin..."

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo "🚀 Starting Ollama..."
ollama serve &
sleep 5

# Pull base model
echo "📥 Downloading Llama 3.2 3B model..."
ollama pull llama3.2:3b

# Create fine-tuned model
echo "🔧 Creating fine-tuned model..."
cd adapters
ollama create texttwin-XXXXXXXXXX -f ../Modelfile.texttwin-XXXXXXXXXX

echo "✅ Setup complete! Run 'python3 texttwin.py XXXXXXXXXX' to start."
""")
    setup_script.chmod(0o755)
    
    # 5. Create Modelfile
    modelfile = package_dir / "Modelfile.texttwin-XXXXXXXXXX"
    modelfile.write_text("""FROM llama3.2:3b
ADAPTER ./adapters/texttwin-XXXXXXXXXX

SYSTEM "You are a text message responder trained on real conversation data. Generate responses that match the user's natural texting style, tone, and patterns."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
""")
    
    # 6. Create run script
    run_script = package_dir / "run.py"
    run_script.write_text("""#!/usr/bin/env python3
import subprocess
import sys
import os

# Start TextTwin for XXXXXXXXXX
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    subprocess.run([sys.executable, "texttwin.py", "XXXXXXXXXX"])
""")
    run_script.chmod(0o755)
    
    # 7. Create installation instructions
    instructions = package_dir / "INSTALL.md"
    instructions.write_text("""# TextTwin Standalone Installation

## Quick Start

1. **Run setup**: `./setup.sh`
2. **Install Python deps**: `pip install -r requirements.txt`  
3. **Start TextTwin**: `python3 run.py`

## What's Included

- ✅ Core TextTwin application
- ✅ Fine-tuned model for contact XXXXXXXXXX
- ✅ Conversation history and RAG databases
- ✅ All required Python dependencies

## Size: ~2.2GB total
- Base model (Llama 3.2 3B): ~1.9GB
- Fine-tuned adapter: ~300MB
- App + data: ~50MB

## Requirements

- macOS or Linux
- Python 3.8+
- 4GB+ RAM
- 3GB+ disk space
""")
    
    print(f"\n🎉 Package created in: {package_dir.absolute()}")
    print(f"📊 Estimated size: ~2.2GB when models are downloaded")
    
    return package_dir

if __name__ == "__main__":
    create_package()