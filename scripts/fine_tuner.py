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
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        # 1) Write training JSONL in chat-message format suitable for tokenizer.apply_chat_template
        prep_task = progress.add_task("📝 Preparing training data...", total=len(training_pairs))
        jsonl_path = Path(f"training_data_{normalized}.jsonl")
        with open(jsonl_path, "w") as f:
            for i, p in enumerate(training_pairs):
                rec = {
                    "messages": [
                        {"role": "system", "content": "You reply in the user's natural texting style."},
                        {"role": "user", "content": p["input"]},
                        {"role": "assistant", "content": p["output"]}
                    ]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                progress.update(prep_task, advance=1)
        
        progress.update(prep_task, description="✅ Training data prepared")
        console.print(f"💾 Created training file: {jsonl_path}")

        # 2) Run LoRA training
        train_task = progress.add_task("🚀 Training LoRA adapter (this may take several minutes)...", total=100)
        base_id = "meta-llama/Llama-3.2-3B-Instruct"
        cmd = f"python3 scripts/train_lora.py --base {base_id} --data {jsonl_path} --out {adapter_dir} --epochs 3 --lr 2e-4"
        
        console.print("🧠 Starting LoRA fine-tuning process...")
        console.print("⏱️  This typically takes 5-15 minutes depending on your hardware")
        
        # Run training with progress simulation
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Simulate progress since we can't easily track real training progress
        import time
        progress_steps = [10, 25, 45, 65, 80, 95, 100]
        step_messages = [
            "Loading model and tokenizer...",
            "Preparing dataset...", 
            "Starting training epoch 1/3...",
            "Training epoch 2/3...",
            "Training epoch 3/3...",
            "Saving adapter...",
            "Training complete!"
        ]
        
        for i, (step, msg) in enumerate(zip(progress_steps, step_messages)):
            time.sleep(2)  # Small delay to show progress
            progress.update(train_task, completed=step, description=f"🚀 {msg}")
            
            # Check if process finished early
            if process.poll() is not None:
                break
        
        # Wait for actual completion
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            progress.update(train_task, description="❌ Training failed!")
            console.print(f"❌ Training failed:\n{stderr}")
            return None
        
        progress.update(train_task, completed=100, description="✅ LoRA training completed!")
        console.print(f"✅ Adapter saved to: {adapter_dir}")

        # 3) Build Modelfile that attaches the adapter
        modelfile_task = progress.add_task("📄 Creating Ollama Modelfile...", total=100)
        
        modelfile_content = f"""FROM llama3.2:3b
ADAPTER ./adapters/{model_name}

SYSTEM \"\"\"You are a text message responder trained on real conversation data. Generate responses that match the user's natural texting style, tone, and patterns.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
        modelfile_path = Path(f"models/Modelfile.{model_name}")
        modelfile_path.parent.mkdir(exist_ok=True)
        modelfile_path.write_text(modelfile_content, encoding="utf-8")
        
        progress.update(modelfile_task, completed=100, description="✅ Modelfile created")

        # 4) Create the Ollama model
        ollama_task = progress.add_task("🏗️ Creating Ollama model with adapter...", total=100)
        
        console.print("🔧 Registering model with Ollama...")
        process = subprocess.Popen(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        # Simulate progress for Ollama model creation
        for step in [20, 40, 60, 80, 100]:
            time.sleep(1)
            progress.update(ollama_task, completed=step)
            if process.poll() is not None:
                break
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            progress.update(ollama_task, description="❌ Ollama model creation failed!")
            console.print(f"❌ Ollama create failed:\n{stderr}")
            return None

        progress.update(ollama_task, completed=100, description="✅ Ollama model created!")

    console.print(f"🎉 Model created successfully: {model_name}")
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