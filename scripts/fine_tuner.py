#!/usr/bin/env python3
"""
Clean fine-tuning system using extracted messages
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from rich.console import Console
from rich.progress import track
from message_extractor import extract_conversation

console = Console()

from datetime import datetime, timedelta

def _collapse_turns(messages, joiner="\n"):
    """Collapse consecutive messages by the same sender into a single 'turn'."""
    turns = []
    current = None
    for m in sorted(messages, key=lambda x: x['timestamp']):
        sender = "you" if m['is_from_me'] else "them"
        if current and current["sender"] == sender:
            current["text"].append(m["text"])
            current["end_ts"] = m["timestamp"]
        else:
            if current:
                current["text"] = joiner.join(current["text"])
                turns.append(current)
            current = {
                "sender": sender,
                "text": [m["text"]],
                "start_ts": m["timestamp"],
                "end_ts": m["timestamp"],
            }
    if current:
        current["text"] = joiner.join(current["text"])
        turns.append(current)
    return turns

def create_training_pairs(messages: list,
                          max_reply_gap_seconds: int = 900,
                          max_context_turns: int = 2) -> list:
    """
    Create training pairs using collapsed turns and a reply window.
    Pairs = (their_turn -> your_next_turn) if your reply arrives within max_reply_gap_seconds.
    Optionally prepend up to `max_context_turns` prior turns as lightweight context.
    """
    turns = _collapse_turns(messages)
    pairs = []
    for i in range(len(turns) - 1):
        a, b = turns[i], turns[i + 1]
        if a["sender"] == "them" and b["sender"] == "you":
            # reply window check
            if (b["start_ts"] - a["end_ts"]) <= max_reply_gap_seconds * 1_000_000_000:
                # build small context window
                ctx_start = max(0, i - max_context_turns)
                context = []
                for t in turns[ctx_start:i]:
                    who = "You" if t["sender"] == "you" else "Them"
                    context.append(f"{who}: {t['text']}")
                context.append(f"Them: {a['text']}")
                pairs.append({
                    "input": "\n".join(context),
                    "output": b["text"],
                })
    return pairs

def create_modelfile(phone_number: str) -> str:
    """Create Ollama Modelfile for fine-tuning"""
    
    normalized = re.sub(r'[^\d]', '', phone_number)
    model_name = f"texttwin-{normalized}"
    
    modelfile_content = f"""FROM llama3.2:3b

SYSTEM \"\"\"You are a text message responder trained on real conversation data. Generate responses that match the user's natural texting style, tone, and patterns. Be conversational, authentic, and match the relationship dynamic shown in the training data.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40"""
    
    return modelfile_content, model_name

def fine_tune_model(phone_number: str, training_pairs: list) -> str:
    """Fine-tune via PEFT LoRA, then create an Ollama model that loads the adapter."""
    import subprocess, shlex
    normalized = re.sub(r'[^\d]', '', phone_number)
    model_name = f"texttwin-{normalized}"
    adapter_dir = Path("adapters") / model_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # 1) Write training JSONL in chat-message format suitable for tokenizer.apply_chat_template
    jsonl_path = Path(f"training_data_{normalized}.jsonl")
    with open(jsonl_path, "w") as f:
        for p in training_pairs:
            rec = {
                "messages": [
                    {"role": "system", "content": "You reply in the user's natural texting style."},
                    {"role": "user", "content": p["input"]},
                    {"role": "assistant", "content": p["output"]}
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    console.print(f"💾 Created training file: {jsonl_path}")

    # 2) Run LoRA training
    base_id = "meta-llama/Llama-3.2-3B-Instruct"
    cmd = f"python3 train_lora.py --base {base_id} --data {jsonl_path} --out {adapter_dir} --epochs 3 --lr 2e-4"
    console.print("🚀 Training LoRA adapter...")
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if res.returncode != 0:
        console.print(f"❌ Training failed:\n{res.stderr}")
        return None
    console.print(f"✅ Adapter saved to: {adapter_dir}")

    # 3) Build Modelfile that attaches the adapter (path is relative to Modelfile)
    modelfile_content = f"""FROM llama3.2:3b
ADAPTER ./adapters/{model_name}

SYSTEM \"\"\"You are a text message responder trained on real conversation data. Generate responses that match the user's natural texting style, tone, and patterns.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
    modelfile_path = Path(f"Modelfile.{model_name}")
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    # 4) Create the Ollama model
    console.print("🏗️ Creating Ollama model with adapter...")
    res = subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        console.print(f"❌ Ollama create failed:\n{res.stderr}")
        return None

    console.print(f"✅ Model created: {model_name}")
    return model_name

def main():
    import sys
    
    if len(sys.argv) != 2:
        console.print("Usage: python3 fine_tuner.py <phone_number>")
        console.print("Example: python3 fine_tuner.py 'XXXXXXXXXX'")
        sys.exit(1)
    
    phone = sys.argv[1]
    
    console.print("🔥 [bold cyan]TextTwin Fine-Tuning System[/bold cyan]")
    console.print("=" * 50)
    
    # Extract messages
    messages = extract_conversation(phone)
    
    if not messages:
        console.print("❌ No messages found. Check phone number format.")
        return
    
    # Create training pairs
    pairs = create_training_pairs(messages)
    
    if len(pairs) < 5:
        console.print(f"⚠️ Only {len(pairs)} training pairs found. Need at least 5 for meaningful fine-tuning.")
        return
    
    # Fine-tune model
    model_name = fine_tune_model(phone, pairs)
    
    if model_name:
        console.print(f"\n🎉 [bold green]Fine-tuning complete![/bold green]")
        console.print(f"Model name: {model_name}")
        console.print(f"Training pairs: {len(pairs)}")
        console.print(f"\nUse with: python3 texttwin.py {phone}")
    else:
        console.print("\n❌ [bold red]Fine-tuning failed[/bold red]")

if __name__ == "__main__":
    main()