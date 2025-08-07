from typing import List, Dict, Optional
import re

def _split_into_sections(text: str):
    # naive section split
    return re.split(r"\n\s*\n", text)

def _heuristic_pairs(text: str) -> List[Dict]:
    pairs = []
    for sec in _split_into_sections(text):
        sec = sec.strip()
        if not sec:
            continue
        # Create 2 QA pairs per section: summary + key-point
        pairs.append({
            "instruction": "次のテキストを要約してください。",
            "input": sec,
            "output": "",
            "tags": ["summary","auto"],
        })
        pairs.append({
            "instruction": "次のテキストの重要ポイントを3つ箇条書きにしてください。",
            "input": sec,
            "output": "",
            "tags": ["keypoints","auto"],
        })
    return pairs

def _llm_assisted_pairs(text: str, model_path: str) -> List[Dict]:
    # Optional: try using a local HF model if provided
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForCausalLM.from_pretrained(model_path)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
        # Prompt to generate Q/A from a section
        pairs = []
        for sec in _split_into_sections(text):
            sec = sec.strip()
            if not sec:
                continue
            prompt = f"与えられた本文から日本語の『質問:』『回答:』形式のQ&Aを2つ作成してください。\n本文:\n{sec}\nQ&A:\n"
            out = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
            # naive parse
            qa = re.findall(r"質問[:：](.+?)\n回答[:：](.+?)(?:\n|$)", out, flags=re.S)
            for q, a in qa[:2]:
                pairs.append({
                    "instruction": q.strip(),
                    "input": "",
                    "output": a.strip(),
                    "tags": ["qa","llm"],
                })
        return pairs if pairs else _heuristic_pairs(text)
    except Exception:
        return _heuristic_pairs(text)

def generate_pairs(text: str, mode: str = "summary_qa", caption_model_path: Optional[str] = None) -> List[Dict]:
    if caption_model_path:
        return _llm_assisted_pairs(text, caption_model_path)
    # heuristic only path
    return _heuristic_pairs(text)
