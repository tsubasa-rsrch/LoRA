import os, json, math
from typing import Dict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

def _load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples

def _format_example(ex, template: str):
    instr = ex.get("instruction","").strip()
    inp = ex.get("input","")
    out = ex.get("output","").strip()
    prompt = template.format(instruction=instr, input=inp)
    return prompt + out

def run_training(cfg: Dict):
    base_model = cfg["base_model"]
    dataset_path = cfg["dataset_path"]
    output_dir = cfg["output_dir"]
    lora_type = cfg.get("lora_type","qlora")
    num_epochs = cfg.get("num_epochs", 1)
    lr = cfg.get("lr", 2e-4)
    batch_size = cfg.get("batch_size", 1)
    grad_accum_steps = cfg.get("grad_accum_steps", 4)
    max_seq_len = cfg.get("max_seq_len", 2048)
    bf16 = cfg.get("bf16", True)
    fp16 = cfg.get("fp16", False)
    template = cfg.get("prompt_template", "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4bit QLoRA setup if requested
    quantization_config = None
    if lora_type == "qlora":
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="bfloat16" if bf16 else "float16",
            )
        except Exception:
            quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype="auto",
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","v_proj","k_proj","o_proj"]  # common for LLaMA-like
    )

    model = get_peft_model(model, lora_cfg)

    # Prepare dataset
    raw = _load_jsonl(dataset_path)

    def tokenize_function(ex):
        text = _format_example(ex, template)
        tokens = tokenizer(text, truncation=True, max_length=max_seq_len)
        return tokens

    # fast local map
    toks = [tokenize_function(ex) for ex in raw]

    class SimpleDataset:
        def __len__(self): return len(toks)
        def __getitem__(self, idx): return toks[idx]

    ds = SimpleDataset()

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoint"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        bf16=bf16,
        fp16=fp16,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    # Save adapter (PEFT)
    adapter_out = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    return {"adapter_path": adapter_out, "logs": os.path.join("outputs","logs","train.log")}
