import os, subprocess, json

def _nvidia_smi_query():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.STDOUT, text=True, timeout=3)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        gpus = []
        for l in lines:
            name, mem = [x.strip() for x in l.split(",")]
            gpus.append({"name": name, "vram_gb": float(mem)/1024.0})
        return gpus
    except Exception:
        return []

def collect_env():
    info = {
        "platform": os.name,
        "cuda_available": False,
        "torch_version": None,
        "gpus": _nvidia_smi_query(),
        "bitsandbytes_available": False,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import bitsandbytes  # noqa: F401
        info["bitsandbytes_available"] = True
    except Exception:
        info["bitsandbytes_available"] = False
    return info
