# TextTwin - Personal Writing Style Cloner 🤖

**TextTwin** is an AI-powered tool that analyzes your personal texting patterns and creates a local AI that can respond exactly like you would. It learns your unique writing style, vocabulary, punctuation habits, and personality to generate authentic responses.

## ✨ Features

- **🔒 100% Private**: All processing happens locally - your messages never leave your machine
- **📱 iMessage Integration**: Safely reads your iMessage database with read-only access
- **🧠 Style Analysis**: Deep linguistic analysis of your texting patterns
- **🤖 Local AI**: Uses Ollama (Llama 3.2) running entirely on your machine  
- **💬 Interactive Chat**: Chat with an AI version of yourself
- **📊 Style Insights**: Detailed reports on your texting personality
- **🎯 High Accuracy**: Mimics your exact punctuation, emoji usage, and phrase preferences

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- macOS (for iMessage integration)
- Homebrew

### Installation

1. **Clone and setup the environment:**
```bash
git clone <this-repo>
cd texttwin
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Install and start Ollama:**
```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
```

3. **Run the demo:**
```bash
python demo.py
```

## 📖 How It Works

### 1. Safe Message Analysis
TextTwin uses multiple safety layers to analyze your messages:
- **Read-only database access** - your original chat.db is never modified
- **Temporary file processing** - works on secure copies only  
- **Hash verification** - ensures original data integrity
- **Permission validation** - confirms safe file access

### 2. Style Pattern Recognition
The AI analyzes:
- **Message length and structure**
- **Vocabulary and phrase patterns**
- **Punctuation and capitalization habits**
- **Emoji usage patterns**
- **Conversation flow and timing**
- **Sentiment and personality indicators**

### 3. Local AI Generation
- Uses **Ollama** with **Llama 3.2 (3B)** model
- Completely local inference - no external API calls
- Custom prompts that encode your specific style
- Context-aware response generation

## 🎮 Usage Examples

### Interactive Chat Mode
```python
from texttwin_engine import TextTwinEngine

engine = TextTwinEngine()
engine.analyze_sample_messages()  # or analyze_imessages()
engine.interactive_chat()
```

### Generate Single Response
```python
engine = TextTwinEngine()
engine.analyze_sample_messages()

response = engine.generate_response("want to grab dinner tonight?")
print(f"You would respond: {response}")
# Output: "yeah sounds good! what time works?"
```

### Batch Processing
```python
engine.batch_responses('input_messages.txt', 'responses.json')
```

## 📁 Project Structure

```
texttwin/
├── safe_imessage_reader.py  # Ultra-safe iMessage database reader
├── message_analyzer.py      # Linguistic pattern analysis engine
├── texttwin_engine.py      # Main AI response generation system
├── demo.py                 # Complete demonstration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛡️ Safety & Privacy

### Data Protection
- **No network access** - all processing is local
- **Read-only operations** - your original messages are never modified
- **Temporary processing** - working copies are automatically cleaned up
- **No data persistence** - messages aren't stored beyond analysis

### Technical Safety Features
- SQLite connections with explicit readonly flags
- File permission verification before access
- Cryptographic hash verification of data integrity
- Automatic cleanup of temporary files
- Comprehensive error handling

## 📊 Style Analysis Report

TextTwin provides detailed insights into your texting style:

```
📱 Your Texting Style Profile
┌─────────────────────────────┬──────────────────┐
│ Average message length      │   28.7 characters│
│ Average words per message   │              5.8 │
│ Capitalization ratio        │            0.35% │
│ Emoji usage rate           │ 0.23 per message │  
│ Unique vocabulary size      │         65 words │
│ Readability score          │         94.1/100 │
└─────────────────────────────┴──────────────────┘
```

## 🔧 Configuration

### Ollama Settings
The system uses these default settings:
- **Model**: llama3.2:3b
- **Temperature**: 0.8 (for natural variation)
- **Max tokens**: 150 (typical text message length)
- **Top-p**: 0.9 (for diverse but focused responses)

### Customization
You can modify parameters in `texttwin_engine.py`:
```python
self.model_name = "llama3.2:3b"  # Change model
temperature = 0.8                 # Adjust creativity
max_tokens = 150                  # Response length limit
```

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
brew services list | grep ollama

# Start Ollama if needed
brew services start ollama

# Verify model is available
ollama list
```

### iMessage Access Issues
- Ensure Terminal has Full Disk Access in System Preferences > Security & Privacy
- Grant permission when prompted for iMessage database access
- Use sample data mode if iMessage access isn't working

### Common Issues
- **"Model not found"**: Run `ollama pull llama3.2:3b`
- **"Permission denied"**: Check file permissions and Full Disk Access
- **"Connection refused"**: Restart Ollama service

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional message sources (WhatsApp, Telegram, etc.)
- More sophisticated style analysis
- GUI interface
- Better error handling
- Performance optimizations

## 📄 License

This project is open source. Please use responsibly and respect privacy.

## ⚠️ Ethical Use

TextTwin is designed for:
- ✅ Personal productivity and fun
- ✅ Understanding your own communication patterns  
- ✅ Privacy-focused AI experimentation
