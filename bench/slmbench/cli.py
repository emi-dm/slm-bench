"""CLI de slmbench. Sin argumentos abre la TUI."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import __version__, config, engine, i18n, leaderboard
from .benchmarks import catalog as cat
from .i18n import T


def _apply(s, args, names):
    for attr, flag in names:
        val = getattr(args, flag, None)
        if val is not None:
            setattr(s, attr, val)


def selected_keys(s, args=None) -> list[str]:
    if args is not None and getattr(args, "benchmarks", None):
        keys = [k.strip() for k in args.benchmarks.split(",") if k.strip()]
        bad = [k for k in keys if k not in cat.catalog()]
        if bad:
            sys.exit(T(f"benchmarks desconocidos: {bad}. Usa `slmbench benchmarks`.", f"unknown benchmarks: {bad}. See `slmbench benchmarks`."))
        return keys
    mode_keys = cat.mode_benchmarks(s.mode)
    return [k for k in (s.benchmarks or mode_keys) if k in mode_keys] or mode_keys


def cmd_setup(args):
    s = config.load()
    s.mode = args.mode
    s.benchmarks = []
    config.save(s)
    r = engine.setup(args.mode, on_line=lambda l: print(l, flush=True), download=not args.no_download)
    print()
    print(T(f"Modo «{args.mode}» listo: {len(r['benchmarks'])} benchmarks.", f"Mode '{args.mode}' ready: {len(r['benchmarks'])} benchmarks."))
    if r["missing_tools"]:
        print(T("No se pueden ejecutar todavía (falta lo indicado):", "Can't run yet (missing what's listed):"))
        for k, tools in r["missing_tools"].items():
            print(f"  · {k}: {', '.join(tools)}")
    if r["failed"]:
        print(T("No se pudieron descargar:", "Could not download:"), ", ".join(r["failed"]))
    print(T("Siguiente paso: slmbench run", "Next: slmbench run"))


def _run_dir(name: str, s) -> Path:
    """A run folder, given as a path or as its name inside the results folder."""
    p = Path(name).expanduser()
    if p.is_dir():
        return p
    q = Path(s.out_dir).expanduser() / name
    if q.is_dir():
        return q
    sys.exit(T(f"No existe la ejecución {name} (ni como ruta ni en {s.out_dir}).", f"No run {name} (neither as a path nor in {s.out_dir})."))


def _print_result(out: Path):
    print("\n" + (out / "report.md").read_text())
    meta = json.loads((out / "summary.json").read_text())["meta"]
    print(f"{T('Guardado en', 'Saved to')} {out}/")
    if meta.get("api_errors"):
        print("⚠ " + T(f"Peticiones fallidas: {meta['api_errors']}", f"Failed requests: {meta['api_errors']}"))
    if meta.get("leaderboard"):
        print(f"{T('Leaderboard actualizado', 'Leaderboard updated')}: {meta['leaderboard']}")
    if meta.get("leaderboard_kept"):
        print(T("Leaderboard: se mantiene el resultado completo anterior de ", "Leaderboard: kept the earlier full result of ")
              + ", ".join(meta["leaderboard_kept"]) + T(" (esta ejecución fue parcial).", " (this run was partial)."))


def cmd_log(args):
    s = config.load()
    print(engine.errors_markdown(_run_dir(args.run, s), tail=args.tail))


def cmd_review(args):
    from . import review
    s = config.load()
    d = _run_dir(args.run, s)
    tasks = review.load(d)
    if not args.task:
        print(review.summary_markdown(d, tasks))
        print("\n" + T("Detalle: slmbench review RUN --task <clave> [--sample N]", "Details: slmbench review RUN --task <key> [--sample N]"))
        return
    t = next((x for x in tasks if x["key"] == args.task or x["task"].split("|")[0] == args.task), None)
    if not t:
        sys.exit(T(f"{args.task} no está en esta ejecución: ", f"{args.task} is not in this run: ") + ", ".join(x["key"] or x["task"] for x in tasks))
    if args.sample is not None:
        smp = next((x for x in t["samples"] if x["index"] == args.sample), None)
        if not smp:
            sys.exit(T("No existe ese problema.", "No such problem."))
        print(review.sample_markdown(t, smp))
        return
    for x in t["samples"]:
        if args.all or x["status"] != "pass":
            first = next((m.get("content", "") for m in x["messages"] if m.get("role") == "user"), "")
            flags = (" truncated" if x["truncated"] else "") + (" empty" if x["empty"] else "")
            print(f"#{x['index']:<4} {x['status']:<7}{flags:<16} " + " ".join(str(first).split())[:90])


def cmd_rescore(args):
    s = config.load()
    d = _run_dir(args.run, s)
    try:
        out = engine.rescore(d, s, on_line=lambda l: print(l, flush=True))
    except RuntimeError as e:
        sys.exit("✗ " + str(e))
    _print_result(out)


def cmd_run(args):
    s = config.load()
    if args.resume:
        d = _run_dir(args.resume, s)
        try:
            out = engine.resume(d, s, on_line=lambda l: print(l, flush=True))
        except RuntimeError as e:
            sys.exit("✗ " + str(e))
        _print_result(out)
        return
    if args.mode:
        s.mode = args.mode
    _apply(s, args, [("base_url", "base_url"), ("model", "model"), ("api_key", "api_key"), ("temperature", "temperature"),
                     ("ld_max_samples", "max_samples"), ("ld_max_new_tokens", "max_tokens"), ("reasoning_budget", "reasoning_budget"),
                     ("workers", "workers"), ("timeout", "timeout"), ("label", "label"), ("out_dir", "out"),
                     ("passk_n", "passk_n"), ("passk_temperature", "passk_temperature")])
    if args.no_leaderboard:
        s.leaderboard_auto = False
    if getattr(args, "label", None):
        s.label_model = "*"                      # --label applies to this run, whatever the model
    keys = selected_keys(s, args)
    specs = engine.specs_for(keys, args.extra if args.extra is not None else s.ld_extra_tasks)
    try:
        engine.check_safe(specs)                 # model code runs only in the sandbox: refuse before starting otherwise
    except RuntimeError as e:
        sys.exit("✗ " + str(e))
    for k in keys:
        miss = cat.missing_tools(k)
        if miss:
            print("⚠ " + T(f"{k}: falta o no funciona {', '.join(miss)} (puntuará 0)", f"{k}: {', '.join(miss)} missing or not working (will score 0)"))
    from .servers import probe
    try:
        ids = probe(s.base_url, s.api_key, timeout=5)
    except Exception as e:
        sys.exit(T(f"No hay servidor en {s.base_url} ({type(e).__name__}). Elige el modelo con `slmbench use` o en la TUI.",
                   f"No server at {s.base_url} ({type(e).__name__}). Pick the model with `slmbench use` or in the TUI."))
    model = s.model or (ids[0] if ids else "default")
    print(f"▶ {T('Modelo', 'Model')}: {model}  ·  {s.base_url}")
    print(f"  {T('Benchmarks', 'Benchmarks')}: {', '.join(keys)}{(' + ' + args.extra) if args.extra else ''}  ·  "
          f"{T('muestras', 'samples')}: {s.ld_max_samples or T('todas', 'all')}\n")
    out = engine.run(s, model, specs, on_line=lambda l: print(l, flush=True))
    _print_result(out)


def cmd_benchmarks(args):
    from .benchmarks import explore
    if args.markdown:
        print(explore.markdown_catalog())
    elif args.key:
        if args.key not in cat.catalog():
            sys.exit(T(f"desconocido: {args.key}", f"unknown: {args.key}"))
        print(explore.card(args.key, sample=not args.no_sample))
    else:
        print(explore.overview(args.mode))


def cmd_chat(args):
    """Interactive chat with a local GGUF via llama-cli / `llama cli` (replaces this process)."""
    import os
    from . import servers
    s = config.load()
    if not servers.cli_command():
        sys.exit(T("No está llama-cli ni `llama cli` en el PATH: ", "Neither llama-cli nor `llama cli` is on the PATH: ") + servers.INSTALL_HINT)
    model = args.model or s.chat_model or (s.srv_hf if s.srv_source == "hf" else s.srv_model)
    if not model:
        found = servers.discover_models(s.model_dirs)
        if not found:
            sys.exit(T("No se encontró ningún .gguf: indica uno (ruta u org/repo:quant).", "No .gguf found: pass one (a path or org/repo:quant)."))
        model = found[0]["path"]
    cmd = servers.chat_command(s, model, args.system if args.system is not None else s.chat_system,
                               args.ctx or s.chat_ctx, s.chat_reasoning_budget if args.reasoning_budget is None else args.reasoning_budget)
    print("$ " + " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


def cmd_models(args):
    from .servers import probe
    s = config.load()
    url = args.base_url or s.base_url
    try:
        ids = probe(url, s.api_key, timeout=5)
    except Exception as e:
        sys.exit(T(f"No hay servidor en {url} ({type(e).__name__}).", f"No server at {url} ({type(e).__name__})."))
    print(f"{T('Modelos en', 'Models at')} {url}:")
    for m in ids:
        mark = "  ←" if m == s.model or (not s.model and m == ids[0]) else ""
        print(f"  • {m}{mark}")
    print("\n" + T("Para elegir uno: slmbench use <modelo> [--base-url URL]", "To pick one: slmbench use <model> [--base-url URL]"))


def cmd_use(args):
    s = config.load()
    _apply(s, args, [("base_url", "base_url"), ("model", "model"), ("label", "label"), ("api_key", "api_key")])
    config.save(s)
    from .servers import probe
    try:
        ids = probe(s.base_url, s.api_key, timeout=5)
        status = T("conectado", "connected") + ("" if (not s.model or s.model in ids) else f" (⚠ {s.model} ∉ {ids})")
    except Exception as e:
        status = T("sin conexión", "not connected") + f" ({type(e).__name__})"
    print(f"{T('Modelo a evaluar', 'Model to evaluate')}: {s.model or T('(el primero del servidor)', '(first one on the server)')}\n"
          f"{T('Servidor', 'Server')}: {s.base_url}  [{status}]")


def cmd_results(args):
    s = config.load()
    for r in engine.list_runs(s.out_dir):
        heads = " · ".join(f"{t.split('|')[0]} {v * 100:.0f}%" for t, v in r["heads"].items())
        print(f"{r['date'][:16]}  {r['label']:<24} {heads}  (n={r['max_samples'] or 'all'})  {r['dir']}")


def cmd_compare(args):
    print(engine.compare(args.runs))


def cmd_leaderboard(args):
    s = config.load()
    if args.action == "show":
        print(leaderboard.text_table(s))
    elif args.action == "rebuild":
        p = leaderboard.rebuild(s)
        print(f"{p['md']}\n{p['html']}\n{p['svg']}")
    elif args.action == "open":
        import webbrowser
        p = leaderboard.rebuild(s)
        webbrowser.open(p["html"].as_uri())
    elif args.action == "path":
        print(leaderboard.paths(s)["md"])
    elif args.action == "remove":
        if not args.id:
            sys.exit("slmbench leaderboard remove <id>   (" + ", ".join(e["id"] for e in leaderboard.load_entries(s)) + ")")
        print("ok" if leaderboard.remove_entry(args.id, s) else f"? {args.id}")


def cmd_config(args):
    s = config.load()
    if args.action == "path":
        print(config.config_path())
    elif args.action == "show":
        d = dataclasses.asdict(s)
        if d.get("api_key"):
            d["api_key"] = "***"
        print(json.dumps(d, indent=1, ensure_ascii=False))
    elif args.action == "set":
        f = {x.name: x for x in dataclasses.fields(s)}
        if args.key not in f or args.value is None:
            sys.exit("slmbench config set <key> <value>   (" + ", ".join(f) + ")")
        cur, v = getattr(s, args.key), args.value
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes", "si", "sí", "on")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        elif isinstance(cur, list):
            v = [x.strip() for x in v.split(",") if x.strip()]
        setattr(s, args.key, v)
        print(config.save(s))
    elif args.action == "reset":
        config.save(config.Settings())


def cmd_tui(args):
    from .tui import run_tui
    run_tui()


def main():
    i18n.set_lang(config.load().ui_lang)
    modes = list(cat.MODES) + ["all"]
    ap = argparse.ArgumentParser(prog="slmbench", description=T(
        "Evaluación de modelos de lenguaje pequeños con benchmarks públicos (vía lighteval). Sin argumentos abre la TUI.",
        "Evaluate small language models on public benchmarks (via lighteval). Without arguments it opens the TUI."))
    ap.add_argument("--version", action="version", version=f"slmbench {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("tui", help=T("interfaz interactiva (por defecto)", "interactive UI (default)")).set_defaults(fn=cmd_tui)

    su = sub.add_parser("setup", help=T("preparar un modo: comprueba herramientas y descarga los datasets", "prepare a mode: check tools and download datasets"))
    su.add_argument("--mode", choices=modes, default="general")
    su.add_argument("--no-download", action="store_true")
    su.set_defaults(fn=cmd_setup)

    rv = sub.add_parser("review", help=T("explorar las respuestas de una ejecución y por qué fallan", "explore a run's answers and why they failed"))
    rv.add_argument("run", help=T("carpeta de la ejecución o su nombre", "run folder or its name"))
    rv.add_argument("--task", help=T("clave del benchmark (p. ej. humaneval_plus)", "benchmark key (e.g. humaneval_plus)"))
    rv.add_argument("--sample", type=int, help=T("número de problema: muestra el detalle", "problem number: show its details"))
    rv.add_argument("--all", action="store_true", help=T("listar también los que aciertan", "also list the ones that pass"))
    rv.set_defaults(fn=cmd_review)
    lg = sub.add_parser("log", help=T("errores de una ejecución (peticiones fallidas, avisos, trazas) y el final del log",
                                      "a run's errors (failed requests, warnings, tracebacks) and the end of its log"))
    lg.add_argument("run", help=T("carpeta de la ejecución o su nombre", "run folder or its name"))
    lg.add_argument("--tail", type=int, default=80, help=T("líneas del final del log", "lines from the end of the log"))
    lg.set_defaults(fn=cmd_log)
    rs = sub.add_parser("rescore", help=T("volver a puntuar una ejecución con sus respuestas guardadas (sin llamar al modelo)",
                                          "score a run again from its saved answers (no calls to the model)"))
    rs.add_argument("run", help=T("carpeta de la ejecución o su nombre", "run folder or its name"))
    rs.set_defaults(fn=cmd_rescore)
    r = sub.add_parser("run", help=T("ejecutar los benchmarks del modo (o los indicados)", "run the mode's benchmarks (or the given ones)"))
    r.add_argument("--mode", choices=modes)
    r.add_argument("--resume", metavar="RUN", help=T("continuar una ejecución interrumpida (reutiliza sus respuestas)",
                                                     "continue an interrupted run (reuses its answers)"))
    r.add_argument("--benchmarks", help=T("claves separadas por comas (ver `slmbench benchmarks`)", "comma-separated keys (see `slmbench benchmarks`)"))
    r.add_argument("--extra", help=T("otras tareas de lighteval, p. ej. 'aime24|0'", "other lighteval tasks, e.g. 'aime24|0'"))
    r.add_argument("--max-samples", type=int, help=T("muestras por benchmark (0 = completo)", "samples per benchmark (0 = full)"))
    r.add_argument("--max-tokens", type=int, help=T("máx. tokens por respuesta", "max tokens per answer"))
    r.add_argument("--reasoning-budget", type=int, help=T("tokens de razonamiento por respuesta: -1 sin límite, 0 sin razonar", "reasoning tokens per answer: -1 unlimited, 0 none"))
    r.add_argument("--temperature", type=float)
    r.add_argument("--passk-n", type=int, help=T("programación: respuestas por problema para pass@1/pass@5 (1 = solo pass@1)",
                                                 "coding: answers per problem for pass@1/pass@5 (1 = pass@1 only)"))
    r.add_argument("--passk-temperature", type=float, help=T("temperatura de esas respuestas (por defecto 0.6)",
                                                             "temperature for those answers (default 0.6)"))
    r.add_argument("--workers", type=int, help=T("peticiones simultáneas (≤ slots del servidor)", "concurrent requests (≤ server slots)"))
    r.add_argument("--base-url")
    r.add_argument("--model")
    r.add_argument("--api-key")
    r.add_argument("--timeout", type=float)
    r.add_argument("--label")
    r.add_argument("--out")
    r.add_argument("--no-leaderboard", action="store_true")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("benchmarks", help=T("benchmarks disponibles, ficha y tarea de muestra", "available benchmarks, card and sample task"))
    b.add_argument("key", nargs="?")
    b.add_argument("--mode", choices=modes)
    b.add_argument("--no-sample", action="store_true")
    b.add_argument("--markdown", action="store_true", help=T("documento con todos los benchmarks", "document with all benchmarks"))
    b.set_defaults(fn=cmd_benchmarks)

    ch = sub.add_parser("chat", help=T("chatear con un GGUF local (llama cli)", "chat with a local GGUF (llama cli)"))
    ch.add_argument("model", nargs="?", help=T("ruta .gguf u org/repo:quant (por defecto, el modelo local que evalúas)",
                                               ".gguf path or org/repo:quant (default: the local model you evaluate)"))
    ch.add_argument("--system", help=T("prompt de sistema", "system prompt"))
    ch.add_argument("--ctx", type=int, help=T("contexto (-c)", "context (-c)"))
    ch.add_argument("--reasoning-budget", type=int, help=T("tokens de razonamiento: -1 sin límite, 0 sin razonar", "reasoning tokens: -1 unlimited, 0 none"))
    ch.set_defaults(fn=cmd_chat)
    mo = sub.add_parser("models", help=T("modelos del servidor configurado", "models on the configured server"))
    mo.add_argument("--base-url")
    mo.set_defaults(fn=cmd_models)

    u = sub.add_parser("use", help=T("fijar el modelo a evaluar", "set the model to evaluate"))
    u.add_argument("model", nargs="?")
    u.add_argument("--base-url")
    u.add_argument("--label")
    u.add_argument("--api-key")
    u.set_defaults(fn=cmd_use)

    sub.add_parser("results", help=T("ejecuciones guardadas", "saved runs")).set_defaults(fn=cmd_results)
    c = sub.add_parser("compare", help=T("comparar ejecuciones", "compare runs"))
    c.add_argument("runs", nargs="+")
    c.set_defaults(fn=cmd_compare)

    lb = sub.add_parser("leaderboard", help="leaderboard.md / .html / .svg")
    lb.add_argument("action", nargs="?", default="show", choices=["show", "rebuild", "open", "path", "remove"])
    lb.add_argument("id", nargs="?")
    lb.set_defaults(fn=cmd_leaderboard)

    k = sub.add_parser("config", help=T("ver o cambiar la configuración", "show or change settings"))
    k.add_argument("action", choices=["show", "set", "path", "reset"])
    k.add_argument("key", nargs="?")
    k.add_argument("value", nargs="?")
    k.set_defaults(fn=cmd_config)

    args = ap.parse_args()
    if not getattr(args, "fn", None):
        if not sys.stdout.isatty():
            ap.print_help()
            return
        args.fn = cmd_tui
    args.fn(args)
