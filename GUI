from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, json, uuid, asyncio
from utils import env_check, dataset, errors
from jobs import caption as caption_job
from jobs import train as train_job
from jobs import eval as eval_job

app = FastAPI(title="LoRA GUI Backend", default_response_class=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PROCESSED = os.path.join("data", "processed")
OUTPUT_ADAPTERS = os.path.join("outputs", "adapters")
OUTPUT_CARDS = os.path.join("outputs", "cards")

class Pair(BaseModel):
    instruction: str
    input: Optional[str] = ""
    output: str
    tags: Optional[List[str]] = []

class IngestBody(BaseModel):
    pairs: List[Pair] = []
    # Optional file-based ingestion is handled via multipart

class CaptionBody(BaseModel):
    text: str
    mode: str = "summary_qa"  # summary_qa / extractive / style
    caption_model_path: Optional[str] = None

class TrainBody(BaseModel):
    base_model: str
    lora_type: str = "qlora" # qlora / lora
    num_epochs: int = 1
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_seq_len: int = 2048
    dataset_path: str = os.path.join(DATA_PROCESSED, "dataset.jsonl")
    output_dir: str = OUTPUT_ADAPTERS
    bf16: bool = True
    fp16: bool = False
    prompt_template: str = "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n"

class EvalBody(BaseModel):
    model_path: Optional[str] = None
    base_model: str
    adapter_path: str
    prompts: List[str]

@app.get("/env")
def get_env():
    info = env_check.collect_env()
    return info

@app.post("/ingest")
async def ingest(body: IngestBody, files: List[UploadFile] = File(None)):
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    all_pairs = [p.model_dump() for p in body.pairs] if body and body.pairs else []

    # Files (CSV/JSONL/MD/TXT/ShareGPT) ingestion
    if files:
        for uf in files:
            content = await uf.read()
            name = uf.filename or f"file_{uuid.uuid4().hex}"
            parsed = dataset.parse_bytes(content, name)
            all_pairs.extend(parsed)

    # Validation
    ok, msgs = dataset.validate_pairs(all_pairs)
    if not ok:
        return {"ok": False, "errors": msgs}

    # Save JSONL
    ds_path = os.path.join(DATA_PROCESSED, "dataset.jsonl")
    with open(ds_path, "w", encoding="utf-8") as f:
        for ex in all_pairs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {"ok": True, "count": len(all_pairs), "path": ds_path}

@app.post("/caption")
def caption(body: CaptionBody):
    pairs = caption_job.generate_pairs(body.text, mode=body.mode, caption_model_path=body.caption_model_path)
    ds_path = os.path.join(DATA_PROCESSED, f"caption_{uuid.uuid4().hex}.jsonl")
    with open(ds_path, "w", encoding="utf-8") as f:
        for ex in pairs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"ok": True, "count": len(pairs), "path": ds_path}

@app.post("/train")
def train(body: TrainBody):
    try:
        result = train_job.run_training(body.model_dump())
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": errors.humanize(str(e))}

@app.post("/eval")
def evaluate(body: EvalBody):
    try:
        result = eval_job.run_eval(body.model_dump())
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": errors.humanize(str(e))}

@app.get("/export/card")
def export_card():
    # generate a minimal model card
    card = {
        "model": "LoRA Adapter",
        "intended_use": "Instruction-tuned conversation/QA/etc.",
        "limitations": ["May hallucinate", "Depends on base model quality"],
        "license": "User-provided data responsibility",
    }
    path = os.path.join(OUTPUT_CARDS, "MODEL_CARD.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    return {"ok": True, "path": path}
