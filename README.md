# TWiN - Text With iMessage Numbers

TWiN analyzes your actual iMessage conversations and generates hyper-personalized responses that match how you text specific contacts.

*name inspired by Mukund Shankar, lots of UI + basic logic heavy lifting done by Claude Code

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama:**
   ```bash
   brew install ollama
   brew services start ollama
   ollama pull llama3.2:3b
   ```

3. **Grant Full Disk Access** (required to read iMessage data):
   - Open System Preferences → Security & Privacy → Privacy
   - Select "Full Disk Access" from left sidebar  
   - Click lock to make changes
   - Add **Terminal** to the list
   - Restart Terminal

4. **Fine-tune model on your conversation:**
   ```bash
   python3 fine_tuner.py "(XXX) XXX-XXXX"
   ```

5. **Use TWiN:**
   ```bash
   python3 texttwin.py "(XXX) XXX-XXXX"
   ```

## 📱 How It Works

### 1) Extract Messages

**Script:** `message_extractor.py`
**Purpose:** Decode the iMessage database and build a clean conversation history.

* Handles **NSKeyedArchiver** format for rich-text messages
* Extracts actual text from `attributedBody` **BLOB** data
* Produces a normalized, readable message log

---

### 2) Fine-tune Model

**Script:** `fine_tuner.py` (or `texttwin-training.ipynb`)

* Builds training pairs from your conversation history
* Trains an **Ollama** adapter on **your** texting style
* Learns your patterns, vocabulary, and response style
* **Alternative:** run `texttwin-training.ipynb` on an A100 GPU (Colab or similar), then copy the resulting adapter into this project

**Place the adapter files like so:**

```
texttwin/
  adapters/
    texttwin-XXXXXXXXXX/
      adapter_config.json
      adapter_model.safetensors
```

**Pull the base model:**

```bash
ollama pull llama3.2:3b
```

**Create a Modelfile that attaches the adapter** (e.g., `Modelfile.texttwin-XXXXXXXXXX`):

```text
FROM llama3.2:3b
ADAPTER ./adapters/texttwin-XXXXXXXXXX

SYSTEM """You are a text message responder trained on real conversation data. Generate responses that match the user's natural texting style, tone, and patterns."""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
```

**Build the model:**

```bash
ollama create texttwin-XXXXXXXXXX -f Modelfile.texttwin-XXXXXXXXXX
```

**Verify it appears:**

```bash
ollama list
# … you should see: texttwin-XXXXXXXXXX:latest
```

---

### 3) Generate Responses

**Script:** `texttwin.py` (interactive chat)

* Uses the fine-tuned model if available
* Falls back to context-based prompting when needed
* Generates responses that sound like you

**Start Ollama:**

```bash
ollama serve
```

Then run your interactive script (example):

```bash
python texttwin.py
```

## 🎯 Features

- 🔥 **True model fine-tuning** - Creates a model that IS you
- ✅ **NSKeyedArchiver decoding** - Handles rich iMessage format
- ✅ **Real conversation extraction** - Uses actual chat history
- ✅ **Interactive chat mode** - Test responses in real-time
- ✅ **Privacy-first** - All processing happens locally
- ✅ **Clean codebase** - 3 simple scripts, minimal dependencies

## 💡 Example Usage

```bash
# Extract and fine-tune
$ python3 fine_tuner.py "(XXX) XXX-XXXX"
📱 Extracting conversation with: (XXX) XXX-XXXX
✅ Extracted 234 messages with text
📈 Your messages: 118
📈 Their messages: 116
✅ Created 89 training pairs
🔥 Fine-tuning model: texttwin-XXXXXXXXXX
✅ Model created successfully: texttwin-XXXXXXXXXX

# Interactive chat
$ python3 texttwin.py "(XXX) XXX-XXXX"
💬 Their message: hey what's up?
🤖 Your response: not much, just chillin. you?
```

## 📁 Clean File Structure

- `message_extractor.py` - Extract and decode iMessage conversations
- `fine_tuner.py` - Fine-tune personalized models 
- `texttwin.py` - Interactive response generation
- `requirements.txt` - Minimal dependencies (rich, requests)
- `old_files/` - Previous versions moved here

## 🔧 Requirements

- macOS (for iMessage access)
- Python 3.11+
- Ollama with llama3.2:3b model
- Full Disk Access permissions

## 🛠️ Troubleshooting

**"Permission denied" error?**
- Grant Full Disk Access to Terminal in System Preferences

**"No messages found" error?**
- Check the available chat identifiers shown in the output
- Try different phone number formats

**"Ollama connection error"?**
- Install Ollama: `brew install ollama`
- Start service: `brew services start ollama`
- Pull model: `ollama pull llama3.2:3b`
