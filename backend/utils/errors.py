def humanize(msg: str) -> str:
    m = msg.lower()
    if "cuda out of memory" in m or "cublas" in m:
        return "GPUメモリ不足です。バッチサイズを下げるかQLoRA/4bitを選択してください。"
    if "bitsandbytes" in m:
        return "bitsandbytesが使えません。インストールまたはCPU/FP16へ切替を検討してください。"
    if "no such file" in m:
        return "パスが正しいか確認してください。ファイルが存在しません。"
    return f"エラー: {msg}"
