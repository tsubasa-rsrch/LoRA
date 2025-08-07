import json, csv, io, re
from typing import List, Dict

def _from_jsonl(text: str) -> List[Dict]:
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pairs.append({
            "instruction": obj.get("instruction","").strip(),
            "input": obj.get("input",""),
            "output": obj.get("output","").strip(),
            "tags": obj.get("tags", []),
        })
    return pairs

def _from_csv(text: str) -> List[Dict]:
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    pairs = []
    for row in reader:
        pairs.append({
            "instruction": (row.get("instruction") or "").strip(),
            "input": row.get("input") or "",
            "output": (row.get("output") or "").strip(),
            "tags": (row.get("tags") or "").split("|") if row.get("tags") else []
        })
    return pairs

def _from_text(name: str, text: str) -> List[Dict]:
    # A very simple heuristic: split by two newlines into chunks and create summary instruction
    chunks = re.split(r"\n\s*\n", text)
    pairs = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        pairs.append({
            "instruction": "次のテキストを要約してください。",
            "input": ch,
            "output": "",
            "tags": ["summary","auto"],
        })
    return pairs

def parse_bytes(content: bytes, filename: str) -> List[Dict]:
    name = filename.lower()
    text = content.decode("utf-8", errors="ignore")
    if name.endswith(".jsonl"):
        return _from_jsonl(text)
    if name.endswith(".csv"):
        return _from_csv(text)
    # sharegpt / md / txt are treated as plain text here
    return _from_text(name, text)

def validate_pairs(pairs: List[Dict]):
    msgs = []
    ok = True
    for i, p in enumerate(pairs):
        if not p.get("instruction"):
            ok = False; msgs.append(f"[{i}] instructionが空です")
        if p.get("output","") == "":
            # allow empty; but warn
            msgs.append(f"[{i}] outputが空です（学習前に自動生成/補完するか、手動で埋めてください）")
    return ok, msgs
