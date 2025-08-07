from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch, os

def run_eval(cfg: Dict):
    base_model = cfg["base_model"]
    adapter_path = cfg["adapter_path"]
    prompts: List[str] = cfg["prompts"]

    tok = AutoTokenizer.from_pretrained(base_model)
    mdl = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    mdl = PeftModel.from_pretrained(mdl, adapter_path)
    mdl = mdl.merge_and_unload()  # faster eval

    pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")

    results = []
    for p in prompts:
        out = pipe(p, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        results.append({"prompt": p, "output": out})
    return {"results": results}
