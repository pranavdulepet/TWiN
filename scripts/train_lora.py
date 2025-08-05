import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from pathlib import Path
import argparse, json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def load_pairs(jsonl_path):
    # Each line: {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_hf_dataset(jsonl_path, tokenizer, chat_template=True):
    raw = load_dataset("json", data_files=jsonl_path, split="train")

    def _to_text(example):
        msgs = example["messages"]
        if chat_template and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        else:
            # Fallback: simple prompt format
            user = next(m["content"] for m in msgs if m["role"]=="user")
            assistant = next(m["content"] for m in msgs if m["role"]=="assistant")
            text = f"<s>[INST]{user}[/INST]\n{assistant}</s>"
        return {"text": text}

    raw = raw.map(_to_text, remove_columns=raw.column_names)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    tokenized = raw.map(_tok, batched=True, remove_columns=raw.column_names)
    # tokenized.set_format(type="torch")
    return tokenized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    console.print("🚀 [bold cyan]LoRA Fine-Tuning Pipeline[/bold cyan]")
    console.print("=" * 50)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        
        # Step 1: Environment setup
        setup_task = progress.add_task("⚙️ Setting up training environment...", total=100)
        
        os.makedirs(args.out, exist_ok=True)
        progress.update(setup_task, advance=20)

        # Device & dtype selection
        use_mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        use_cuda = torch.cuda.is_available()

        if use_mps:
            device = "mps"
            load_dtype = torch.float16
            device_name = "Apple Silicon (MPS)"
        elif use_cuda:
            device = "cuda"
            load_dtype = torch.bfloat16
            device_name = f"CUDA GPU ({torch.cuda.get_device_name()})"
        else:
            device = "cpu"
            load_dtype = torch.float32
            device_name = "CPU"

        progress.update(setup_task, advance=30, description=f"🖥️ Using device: {device_name}")
        console.print(f"🖥️ Training device: {device_name}")
        
        progress.update(setup_task, advance=50, description="📚 Loading tokenizer...")

        # Step 2: Load tokenizer
        tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        progress.update(setup_task, completed=100, description="✅ Environment setup complete")

        # Step 3: Load base model
        model_task = progress.add_task(f"🧠 Loading base model: {args.base}...", total=100)
        
        console.print(f"📥 Downloading/loading model: {args.base}")
        console.print("⏱️  This may take several minutes for first-time download...")
        
        attn_impl = "sdpa"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base,
                torch_dtype=load_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
            progress.update(model_task, advance=70)
        except Exception as e:
            console.print(f"⚠️  SDPA attention failed, falling back to eager: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base,
                torch_dtype=load_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            progress.update(model_task, advance=70)

        if device != "cpu":
            progress.update(model_task, advance=20, description=f"🚚 Moving model to {device}...")
            model.to(device)

        progress.update(model_task, completed=100, description="✅ Base model loaded")

        # Step 4: Apply LoRA configuration
        lora_task = progress.add_task("🔧 Applying LoRA configuration...", total=100)
        
        lcfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        )
        progress.update(lora_task, advance=50)
        
        model = get_peft_model(model, lcfg)
        progress.update(lora_task, completed=100, description="✅ LoRA adapter applied")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        console.print(f"🎯 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # Step 5: Prepare dataset
        data_task = progress.add_task("📊 Preparing training dataset...", total=100)
        
        ds = build_hf_dataset(args.data, tok, chat_template=True)
        progress.update(data_task, completed=100, description=f"✅ Dataset ready: {len(ds)} samples")
        
        console.print(f"📈 Training samples: {len(ds)}")

        # Step 6: Configure training
        config_task = progress.add_task("⚙️ Configuring training parameters...", total=100)
        
        targs = TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bsz,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            logging_steps=25,
            save_strategy="epoch",
            report_to=[],
            no_cuda=(device != "cuda"),
            group_by_length=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=2,
            optim="adamw_torch",
        )
        
        progress.update(config_task, advance=50)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=16)
        trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=data_collator)
        
        progress.update(config_task, completed=100, description="✅ Training configuration ready")
        
        # Step 7: Start training
        training_task = progress.add_task(f"🎯 Training for {args.epochs} epochs...", total=100)
        
        console.print(f"🚀 Starting training for {args.epochs} epochs...")
        console.print(f"📊 Effective batch size: {args.bsz * args.grad_accum}")
        console.print(f"📈 Learning rate: {args.lr}")
        console.print("⏱️  Training typically takes 5-15 minutes depending on dataset size and hardware")
        
        # Training happens here - we can't easily track real progress without modifying Trainer
        # So we'll just show that training is in progress
        progress.update(training_task, advance=10, description="🎯 Training in progress...")
        
        try:
            trainer.train()
            progress.update(training_task, completed=90, description="🎯 Training completed, saving model...")
        except Exception as e:
            progress.update(training_task, description="❌ Training failed!")
            console.print(f"❌ Training failed: {e}")
            raise
        
        # Step 8: Save model
        save_task = progress.add_task("💾 Saving LoRA adapter...", total=100)
        
        model.save_pretrained(args.out)
        progress.update(save_task, completed=100, description="✅ LoRA adapter saved")
        
        progress.update(training_task, completed=100, description="✅ Training pipeline completed!")

    console.print(f"🎉 [bold green]LoRA fine-tuning completed successfully![/bold green]")
    console.print(f"📁 Adapter saved to: {args.out}")
    console.print(f"🎯 Trainable parameters: {trainable_params:,}")
    console.print("🚀 Ready to create Ollama model with this adapter!")

if __name__ == "__main__":
    main()
