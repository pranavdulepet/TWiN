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

    os.makedirs(args.out, exist_ok=True)

    # --- Device & dtype selection ---
    use_mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    if use_mps:
        device = "mps"
        load_dtype = torch.float16         # OK for weights on MPS
    elif use_cuda:
        device = "cuda"
        load_dtype = torch.bfloat16        # or torch.float16 if you prefer
    else:
        device = "cpu"
        load_dtype = torch.float32


    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn_impl = "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",   # fallback if sdpa not viable
        )

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.base,
    #     torch_dtype=load_dtype,
    #     low_cpu_mem_usage=True,
    #     attn_implementation="eager",   # safest on MPS
    # )
    if device != "cpu":
        model.to(device)

    # LoRA config (tweak as you like)
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lcfg)

    ds = build_hf_dataset(args.data, tok, chat_template=True)

    # flags for accelerate
    fp16_flag = False         # <- IMPORTANT on MPS
    bf16_flag = False

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=3,
        per_device_train_batch_size=2,    # works with checkpointing on MPS
        gradient_accumulation_steps=8,    # effective batch = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
        no_cuda=True,                     # we're on MPS
        group_by_length=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        optim="adamw_torch",
    )

    def data_collator(features):
        # Causal LM labels = input_ids (shifted in model)
        import torch
        pad = tok.pad_token_id
        input_ids = [f["input_ids"] for f in features]
        attn = [f["attention_mask"] for f in features]
        max_len = max(len(x) for x in input_ids)
        def pad_to(x, pad_id):
            return x + [pad_id] * (max_len - len(x))
        input_ids = [pad_to(x, pad) for x in input_ids]
        attn = [pad_to(x, 0) for x in attn]
        input_ids = torch.tensor(input_ids)
        attn = torch.tensor(attn)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=16)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(args.out)  # saves LoRA adapter (PEFT format)

if __name__ == "__main__":
    main()
