"""Unsloth Studio as the model's server, with LoRA adapters.

Studio serves GGUF models through its own llama-server and loads them through its API: `/v1/load` takes the model, its
quantization and `llama_extra_args`, arguments it passes to llama-server as they are, so an adapter goes in as
`--lora <file>` (or `--lora-scaled <file>:<scale>`), the same as with a llama-server started by slmbench. The adapter is
given like everywhere else in slmbench (a .gguf, org/repo[:file.gguf], a Hub URL; a PEFT repo uses its -GGUF version,
lora.py) and downloaded to the Hugging Face cache, which Studio can read because it runs on this machine.

Studio's other way, loading a PEFT adapter itself (`is_lora`), goes through its MLX backend, which can't apply LFM3
adapters yet; slmbench uses the GGUF one. Studio always asks for an API key: servers.effective_key.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .i18n import T


def _root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def _call(base_url: str, key: str, path: str, body: dict | None = None, timeout: float = 10):
    req = urllib.request.Request(_root(base_url) + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def models(base_url: str, key: str) -> list[dict]:
    """The models Studio has downloaded: [{id, quant, loaded}]."""
    return [{"id": m.get("id"), "quant": m.get("quant") or "", "loaded": bool(m.get("loaded"))}
            for m in (_call(base_url, key, "/v1/models").get("data") or []) if m.get("id")]


def status(base_url: str, key: str) -> dict:
    """What Studio has loaded: {model, lora} (lora: the adapter file(s) in its llama-server arguments)."""
    try:
        st = _call(base_url, key, "/v1/status")
    except Exception:  # noqa: BLE001
        return {}
    from .servers import _studio_lora
    info = _studio_lora(st)
    return {"model": st.get("active_model") or "", "lora": info.get("lora", ""), "variant": st.get("gguf_variant") or ""}


def _local(base_url: str) -> bool:
    return bool(re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?(/|$)", base_url))


def load(base_url: str, key: str, model: str, variant: str = "", adapter: str = "", scale: float = 1.0) -> dict:
    """Loads `model` (a GGUF) in Studio, with the LoRA adapter if given. Returns {model, lora, context_length}; raises
    RuntimeError with Studio's own message when the load fails."""
    from . import lora
    if not model:
        raise RuntimeError(T("Elige el modelo base de Studio.", "Pick Studio's base model."))
    # always send the arguments: Studio keeps the previous load's (an earlier --lora too) when a request leaves them out
    body: dict = {"model_path": model, "llama_extra_args": []}
    if variant:
        body["gguf_variant"] = variant
    source, want = "", ""
    if (adapter or "").strip():
        if not _local(base_url):
            raise RuntimeError(T("El adaptador se descarga en este equipo y Studio tiene que poder leerlo: con un Studio remoto, "
                                 "cárgalo desde el propio Studio.",
                                 "The adapter is downloaded on this machine and Studio has to read it: with a remote Studio, load "
                                 "it from Studio itself."))
        info = lora.resolve(adapter)
        body["llama_extra_args"] = lora.flags(info["path"], scale or 1.0)
        source, want = info["source"], info["path"].rsplit("/", 1)[-1]
    cur = status(base_url, key)
    if cur.get("model") == model and (cur.get("lora") or "").split(" ×")[0] != want:
        body["force_reload"] = True                    # same model, another adapter (or none): a fresh llama-server
    try:
        r = _call(base_url, key, "/v1/load", body, timeout=1800)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail") or detail
        except ValueError:
            pass
        raise RuntimeError(f"Unsloth Studio ({e.code}): {str(detail)[:400]}") from e
    err = r.get("_deferred_error") or ({"detail": r.get("detail")} if r.get("detail") and not r.get("status") else None)
    if err:
        raise RuntimeError("Unsloth Studio: " + str(err.get("detail") or err)[:400])
    now = status(base_url, key)                       # what Studio really runs: refuse to report another adapter
    got = (now.get("lora") or "").split(" ×")[0]
    if got != want:
        raise RuntimeError(T(f"Studio cargó {now.get('model') or model} con el adaptador «{got or 'ninguno'}» en vez de "
                             f"«{want or 'ninguno'}».", f"Studio loaded {now.get('model') or model} with adapter '{got or 'none'}' "
                             f"instead of '{want or 'none'}'."))
    return {"model": r.get("model") or model, "lora": source, "context_length": r.get("context_length"),
            "extra": r.get("requested_llama_extra_args") or []}
