"""LoRA adapters for the model served by llama-server (`--lora`).

What can be given (Settings.srv_lora, the TUI's "LoRA adapter" field):
- a local adapter file: `~/models/LFM3-1.7B-harbor-opencode-lora-F16.gguf`;
- a Hugging Face repo with a GGUF adapter: `harness-hub/LFM3-1.7B-opencode-GGUF` (the adapter file is found in it; with
  several, the F16 one, else the first), or `org/repo:file.gguf` to pick one;
- the URL of a repo or file on the Hub: `https://huggingface.co/harness-hub/LFM3-1.7B-opencode`, `…/blob/main/x.gguf`;
- a PEFT adapter repo (adapter_model.safetensors, which llama.cpp can't load): its `-GGUF` sibling is used when it
  exists (`harness-hub/LFM3-1.7B-opencode` -> `harness-hub/LFM3-1.7B-opencode-GGUF`); otherwise the error says how to
  convert it (llama.cpp's convert_lora_to_gguf.py).

Files from the Hub go to the Hugging Face cache, like the models llama-server downloads with -hf. A model served with an
adapter is a different model: its default alias (the leaderboard row) is the adapter's name, and the adapter is recorded
with every run (llama-server's /lora-adapters).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .i18n import T

_HF_URL = re.compile(r"^https?://(?:www\.)?huggingface\.co/([^/\s]+/[^/\s?#]+)(?:/(?:blob|resolve)/[^/]+/(.+?))?/?(?:[?#].*)?$")


def parse(spec: str) -> dict:
    """{"kind": "file", "path"} or {"kind": "hf", "repo", "file" (None = find it)}; raises ValueError."""
    spec = (spec or "").strip().strip("'\"")
    if not spec:
        raise ValueError(T("No se ha indicado ningún adaptador LoRA.", "No LoRA adapter given."))
    m = _HF_URL.match(spec)
    if m:
        return {"kind": "hf", "repo": m.group(1), "file": m.group(2)}
    p = Path(os.path.expanduser(spec))
    repo, _, file = spec.partition(":")
    hub = re.fullmatch(r"[\w.-]+/[\w.-]+", repo) and not spec.startswith(("/", "~", "."))
    if p.exists() or spec.startswith(("/", "~", ".")) or (p.suffix.lower() == ".gguf" and not hub):
        return {"kind": "file", "path": str(p)}
    if hub:
        return {"kind": "hf", "repo": repo, "file": file or None}
    raise ValueError(T(f"No reconozco «{spec}» como adaptador: usa un fichero .gguf, org/repo[:fichero.gguf] o la URL del Hub.",
                       f"'{spec}' isn't an adapter: use a .gguf file, org/repo[:file.gguf] or a Hub URL."))


def name(spec: str) -> str:
    """The adapter's name, without network: the repo (without -GGUF) or the file's stem. Used as the default alias."""
    try:
        p = parse(spec)
    except ValueError:
        return ""
    if p["kind"] == "file":
        return re.sub(r"(?i)[-_.](lora[-_.])?(f16|bf16|f32|q8_0)$", "", Path(p["path"]).stem)
    return re.sub(r"(?i)[-_.]gguf$", "", p["repo"])


def _gguf_files(repo: str) -> list[str]:
    from huggingface_hub import HfApi
    return sorted(f for f in HfApi().list_repo_files(repo) if f.lower().endswith(".gguf"))


def _adapters(files: list[str]) -> list[str]:
    """The .gguf files of a repo that are LoRA adapters by their name: a repo's other .gguf files are whole models, and
    passing a model as --lora would fail (or worse)."""
    return [f for f in files if re.search(r"(?i)lora|adapter", Path(f).name)]


def _pick(files: list[str]) -> str:
    """The adapter file: F16 first, then BF16, then the first."""
    pool = files
    for want in ("f16", "bf16"):
        hit = [f for f in pool if re.search(rf"(?i)[-_.]{want}\.gguf$", f)]
        if hit:
            return hit[0]
    return pool[0]


def find(spec: str) -> dict:
    """Where the adapter is, without downloading it: {"kind", "path"} or {"kind", "repo", "file"}. Raises RuntimeError."""
    p = parse(spec)
    if p["kind"] == "file":
        if not os.path.isfile(p["path"]):
            raise RuntimeError(T(f"No existe el fichero {p['path']}.", f"{p['path']} doesn't exist."))
        return p
    if p["file"]:
        return p
    try:
        files = _gguf_files(p["repo"])
    except Exception as e:  # noqa: BLE001 - network, a typo, a private repo
        raise RuntimeError(T(f"No se pudo leer {p['repo']} en Hugging Face: {e}", f"Could not read {p['repo']} on Hugging Face: {e}")) from e
    if _adapters(files):
        return {**p, "file": _pick(_adapters(files))}
    if files:
        raise RuntimeError(T(f"Los .gguf de {p['repo']} son modelos completos, no adaptadores LoRA. Indica el fichero del adaptador "
                             "(org/repo:fichero.gguf) si estás seguro de que lo es.",
                             f"The .gguf files of {p['repo']} are whole models, not LoRA adapters. Give the adapter's file "
                             "(org/repo:file.gguf) if you're sure it is one."))
    sibling = p["repo"] + "-GGUF"                      # a PEFT repo: its GGUF conversion, published next to it
    try:
        files = _adapters(_gguf_files(sibling))
    except Exception:  # noqa: BLE001 - it doesn't exist
        files = []
    if files:
        return {"kind": "hf", "repo": sibling, "file": _pick(files), "from": p["repo"]}
    raise RuntimeError(T(
        f"No hay ningún adaptador LoRA en GGUF ni en {p['repo']} ni en {sibling}, y llama.cpp solo carga adaptadores GGUF. "
        "Si es un adaptador PEFT (adapter_model.safetensors), conviértelo con convert_lora_to_gguf.py de llama.cpp e indica "
        "el fichero resultante.",
        f"There's no GGUF LoRA adapter in {p['repo']} or in {sibling}, and llama.cpp only loads GGUF adapters. If it's a PEFT "
        "adapter (adapter_model.safetensors), convert it with llama.cpp's convert_lora_to_gguf.py and give the resulting file."))


def resolve(spec: str) -> dict:
    """The adapter as a local file (downloaded to the Hugging Face cache if needed): {"path", "name", "source"}."""
    where = find(spec)
    if where["kind"] == "file":
        return {"path": where["path"], "name": name(spec), "source": where["path"]}
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(where["repo"], where["file"])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(T(f"No se pudo descargar {where['repo']}/{where['file']}: {e}",
                             f"Could not download {where['repo']}/{where['file']}: {e}")) from e
    return {"path": path, "name": re.sub(r"(?i)[-_.]gguf$", "", where["repo"]), "source": f"{where['repo']}:{where['file']}",
            "repo": where["repo"]}


def repo_of(path: str) -> str:
    """The Hub repo an adapter file came from, if it sits in the Hugging Face cache (models--org--repo/…); else ""."""
    part = next((x for x in Path(str(path)).parts if x.startswith("models--")), "")
    return part[8:].replace("--", "/") if part else ""


def base_model(repo: str, hops: int = 4) -> str:
    """The model an adapter repo applies to, following its cards' base_model while it is itself an adapter:
    harness-hub/LFM3-1.7B-opencode-GGUF -> harness-hub/LFM3-1.7B-opencode -> LiquidAI/LFM3-1.7B. "" if unknown."""
    from huggingface_hub import HfApi
    cur = repo
    for _ in range(hops):
        try:
            info = HfApi().model_info(cur, timeout=10)
        except Exception:  # noqa: BLE001 - offline, private, gone
            return "" if cur == repo else cur
        tags = info.tags or []
        if not any(t.startswith("base_model:adapter:") for t in tags):
            return "" if cur == repo else cur        # not an adapter: this is the base
        base = (info.card_data or {}).get("base_model") if info.card_data else None
        base = base[0] if isinstance(base, list) and base else base
        if not base:
            return ""
        cur = base
    return cur


def flags(path: str, scale: float = 1.0) -> list[str]:
    """llama-server's arguments for one adapter."""
    return ["--lora", path] if float(scale) == 1.0 else ["--lora-scaled", f"{path}:{float(scale):g}"]
