"""Benchmarks de programación para lighteval (`--custom-tasks`): HumanEval+, MBPP+, CRUXEval (O/I) y MultiPL-E (JS, C++, Java, Rust, Go).

Todos son de generación con chat: el modelo devuelve código en un bloque markdown, se extrae el bloque más largo del lenguaje
objetivo (si no hay vallas, el texto entero) y se ejecuta contra los tests del dataset en un subproceso con límites de tiempo/CPU.
Métrica: pass@1 (una generación por problema, 1 si pasan todos los tests).

- humaneval_plus  evalplus/humanevalplus (164): función + `test` + `check(entry_point)`.
- mbpp_plus       evalplus/mbppplus (378): función + `test_imports` + `test`.
- cruxeval_o/_i   cruxeval-org/cruxeval (800): predecir la salida (`f(input) == ??`) o una entrada (`f(??) == output`);
                  se comprueba ejecutando la aserción con el código dado.
- multiple_*      nuprl/MultiPL-E humaneval-{js,cpp,java,rs,go}: el programa es la función completada + `tests`
                  (como MultiPL-E: prompt + completion + tests), compilado/ejecutado con node, clang++/g++, javac+java, rustc, go.

La ejecución NO es un sandbox de seguridad completo (subproceso, directorio temporal, límites de CPU/memoria, sin stdin).
Si falta una herramienta (p. ej. `go`) la muestra puntúa 0 y se avisa una vez en el log.
"""
from __future__ import annotations

import ast
import logging
import os
import re

import numpy as np
from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

from slmbench.checks import run_python

logger = logging.getLogger(__name__)

PY_TIMEOUT = 20.0            # EvalPlus/MBPP+ tienen tests pesados
CRUX_TIMEOUT = 10.0
COMPILE_TIMEOUT = 30.0
RUN_TIMEOUT = 20.0

# ======================================================================================
# Extracción de código
# ======================================================================================
_FENCE_RE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)

LANG_ALIASES = {
    "python": {"python", "py", "python3"},
    "js": {"javascript", "js", "node", "nodejs", "jsx"},
    "cpp": {"cpp", "c++", "cc", "cxx", "hpp", "c"},
    "java": {"java"},
    "rs": {"rust", "rs"},
    "go": {"go", "golang"},
}


def _strip_think(text: str) -> str:
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text


def extract_code(text: str, lang: str) -> str:
    """Bloque vallado más largo en el lenguaje (empate: el último); si no hay, el bloque sin etiqueta más largo; si no, todo el texto."""
    text = _strip_think(text)
    blocks = _FENCE_RE.findall(text)
    aliases = LANG_ALIASES.get(lang, {lang})
    for cand in ([b for l, b in blocks if l.lower() in aliases], [b for l, b in blocks if not l]):
        if cand:
            best = max(range(len(cand)), key=lambda i: (len(cand[i].strip()), i))
            return cand[best]
    return text.strip()


# ======================================================================================
# Ejecución de programas no Python
# ======================================================================================
_warned: set[str] = set()


# ---------- utilidades de llaves (para lenguajes con { }) ----------
_CHAR_RE = re.compile(r"'(?:\\[^']{1,10}|[^'\\\n])'")


def _scan(code: str, lang: str, start: int = 0):
    """Recorre `code` desde `start` saltando comentarios y literales; produce (índice, carácter) de llaves."""
    i, n = start, len(code)
    while i < n:
        c = code[i]
        if code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j
            continue
        if code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"' or (c == "'" and lang == "js"):
            j = i + 1
            while j < n and code[j] != c:
                if code[j] == "\\":
                    j += 1
                elif code[j] == "\n" and lang != "rs":
                    break
                j += 1
            i = j + 1
            continue
        if c == "`" and lang in ("js", "go"):
            j = code.find("`", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == "'":
            m = _CHAR_RE.match(code, i)
            i = m.end() if m else i + 1
            continue
        if c in "{}":
            yield i, c
        i += 1


def _match_brace(code: str, open_idx: int, lang: str) -> int:
    """Índice de la llave que cierra la abierta en `open_idx` (o len(code) si no se cierra)."""
    depth = 0
    for i, c in _scan(code, lang, open_idx):
        depth += 1 if c == "{" else -1
        if depth == 0:
            return i
    return len(code)


def _close_index(body: str, lang: str) -> int | None:
    """En un cuerpo de función (con la `{` ya abierta en el prompt), índice de la `}` que la cierra, o None."""
    depth = 0
    for i, c in _scan(body, lang):
        depth += 1 if c == "{" else -1
        if depth < 0:
            return i
    return None


def _remove_blocks(code: str, pattern: str, lang: str) -> str:
    """Elimina las definiciones que casan con `pattern` (desde el inicio de su línea hasta su llave de cierre)."""
    rx = re.compile(pattern, re.M)
    while True:
        m = rx.search(code)
        if not m:
            return code
        ob = code.find("{", m.end())
        if ob < 0:
            return code[:m.start()]
        end = _match_brace(code, ob, lang)
        code = code[:m.start()] + code[end + 1:]


def _toplevel_spans(code: str, lang: str):
    """(inicio, llave_abre, llave_cierra) de los bloques de nivel superior."""
    spans, depth, open_at = [], 0, None
    for i, c in _scan(code, lang):
        if c == "{":
            if depth == 0:
                open_at = i
            depth += 1
        else:
            depth -= 1
            if depth == 0 and open_at is not None:
                spans.append((open_at, i))
            depth = max(depth, 0)
    return spans


# ======================================================================================
# HumanEval+ / MBPP+
# ======================================================================================
_STDIN_GUARD = "import sys as _slm_sys, io as _slm_io\n_slm_sys.stdin = _slm_io.StringIO('')\n"
_PY_IMPORT_RE = re.compile(r"^(?:import \S.*|from \S+ import .*)$", re.M)

HUMANEVAL_INSTR = ("Complete the following Python function. Return the complete function (signature included, plus any imports or "
                   "helper functions it needs) in a single ```python code block.\n\n```python\n{prompt}```")
MBPP_INSTR = ("{prompt}\nYour code should pass this test:\n\n{test}\n\nWrite the Python function and return it in a single "
              "```python code block.")


def humaneval_plus_prompt(line: dict, task_name: str = "humaneval_plus") -> Doc:
    return Doc(task_name=task_name, query=HUMANEVAL_INSTR.format(prompt=line["prompt"]), choices=[""], gold_index=0,
               specific={"prompt": line["prompt"], "test": line["test"], "entry_point": line["entry_point"],
                         "task_id": line["task_id"]})


def mbpp_plus_prompt(line: dict, task_name: str = "mbpp_plus") -> Doc:
    first = (line.get("test_list") or [""])[0]
    return Doc(task_name=task_name, query=MBPP_INSTR.format(prompt=line["prompt"].strip(), test=first), choices=[""], gold_index=0,
               specific={"test": line["test"], "test_imports": list(line.get("test_imports") or []), "task_id": line["task_id"]})


def score_humaneval(code: str, spec: dict) -> tuple[bool, str]:
    entry = spec["entry_point"]
    if not re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry)}\s*\(", code, re.M):
        code = spec["prompt"] + code              # el modelo devolvió solo el cuerpo
    imports = "\n".join(_PY_IMPORT_RE.findall(spec["prompt"]))
    program = _STDIN_GUARD + imports + "\n" + code
    # HumanEval/32 (find_zero): el test del dataset de HF tiene los argumentos de _poly invertidos (falla con cualquier
    # solución, también la canónica). Se corrige: vale una raíz con |poly(x)| <= 1e-4 o el mismo valor que la referencia; las
    # 6 de 886 entradas mal condicionadas en que la propia referencia no es raíz no se exigen.
    test = spec["test"].replace("assert _poly(*candidate(*inp), inp) <= 0.0001",
                                "_o = candidate(*inp); assert abs(_poly(*inp, _o)) <= 0.0001 or abs(_o - exp) <= 1e-6 "
                                "or abs(_poly(*inp, exp)) > 0.0001")
    return run_python(program, test + f"\n\ncheck({entry})\n", timeout=PY_TIMEOUT)


def score_mbpp(code: str, spec: dict) -> tuple[bool, str]:
    program = _STDIN_GUARD + code
    tests = "\n".join(spec["test_imports"]) + "\n\n" + spec["test"]
    return run_python(program, tests, timeout=PY_TIMEOUT)


# ======================================================================================
# CRUXEval
# ======================================================================================
CRUX_O_INSTR = (
    "You are given a Python function and an assertion containing an input to the function. Find the output of executing the "
    "function on the given input, even if the function is incorrect or incomplete.\n\n```python\n{code}\nassert f({input}) == ??\n```\n\n"
    "You may reason step by step. Then give the output as a Python literal (no expressions or function calls) between "
    "[ANSWER] and [/ANSWER], for example: [ANSWER]'abc'[/ANSWER]")
CRUX_I_INSTR = (
    "You are given a Python function and an output. Find any input such that executing the function on that input leads to the "
    "given output. There may be several answers; give only one.\n\n```python\n{code}\nassert f(??) == {output}\n```\n\n"
    "You may reason step by step. Then give the completed assertion between [ANSWER] and [/ANSWER], for example: "
    "[ANSWER]assert f(1, 'a') == {output}[/ANSWER]")


def cruxeval_o_prompt(line: dict, task_name: str = "cruxeval_o") -> Doc:
    return Doc(task_name=task_name, query=CRUX_O_INSTR.format(code=line["code"], input=line["input"]), choices=[""], gold_index=0,
               specific={"code": line["code"], "input": line["input"], "output": line["output"], "id": line["id"]})


def cruxeval_i_prompt(line: dict, task_name: str = "cruxeval_i") -> Doc:
    return Doc(task_name=task_name, query=CRUX_I_INSTR.format(code=line["code"], output=line["output"]), choices=[""], gold_index=0,
               specific={"code": line["code"], "input": line["input"], "output": line["output"], "id": line["id"]})


_ANSWER_RE = re.compile(r"\[ANSWER\](.*?)(?:\[/ANSWER\]|\Z)", re.S | re.I)


def extract_crux_answer(text: str) -> str:
    text = _strip_think(text)
    found = _ANSWER_RE.findall(text)
    if found:
        ans = found[-1].strip()
    else:
        blocks = [b for l, b in _FENCE_RE.findall(text) if l.lower() in ("python", "py", "")]
        if blocks:
            ans = blocks[-1].strip()
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            asserts = [l for l in lines if l.startswith("assert ")]
            ans = asserts[-1] if asserts else (lines[-1] if lines else "")
    m = re.fullmatch(r"```[\w]*\s*\n?(.*?)\n?```", ans, re.S)          # valla dentro de las etiquetas
    if m:
        ans = m.group(1).strip()
    return ans.strip().strip("`").strip()


def _is_f_call(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "f"


def parse_crux_answer(ans: str, mode: str) -> tuple[str, str] | None:
    """mode 'o': ('value', literal). mode 'i': ('call', 'f(...)') o ('args', '...'). None si no se puede interpretar."""
    if not ans:
        return None
    try:
        tree = ast.parse(ans)
        stmts = [s for s in tree.body if isinstance(s, (ast.Assert, ast.Expr))]
    except SyntaxError:
        tree, stmts = None, []
    if stmts:
        st = stmts[-1]
        seg = lambda n: ast.get_source_segment(ans, n)
        node = st.test if isinstance(st, ast.Assert) else st.value
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            left, right = node.left, node.comparators[0]
            call, other = (left, right) if _is_f_call(left) else (right, left) if _is_f_call(right) else (None, right)
            if mode == "o":
                node = other
            else:
                return ("call", seg(call)) if call is not None else None
        if mode == "o":
            if any(_is_f_call(n) for n in ast.walk(node)):
                return None                                     # repetir la llamada no vale como predicción
            return ("value", seg(node))
        return ("call", seg(node)) if _is_f_call(node) else ("args", seg(node))
    if mode == "o":
        return None if re.search(r"\bf\s*\(", ans) else ("value", ans)
    m = re.match(r"(?:assert\s+)?(f\s*\(.*\))\s*(?:==.*)?$", ans, re.S)
    return ("call", m.group(1)) if m else ("args", ans)


def score_crux(answer_text: str, spec: dict, mode: str) -> tuple[bool, str]:
    parsed = parse_crux_answer(extract_crux_answer(answer_text), mode)
    if not parsed or not parsed[1]:
        return False, "no answer"
    kind, val = parsed
    if mode == "o":
        test = f"assert f({spec['input']}) == (\n{val}\n)"
    elif kind == "call":
        test = f"assert (\n{val}\n) == ({spec['output']})"
    else:
        test = f"assert f(\n{val}\n) == ({spec['output']})"
    return run_python(_STDIN_GUARD + spec["code"], test, timeout=CRUX_TIMEOUT)


# ======================================================================================
# MultiPL-E
# ======================================================================================
from slmbench.benchmarks.code_catalog import CATALOG, MULTIPLE_LANGS  # noqa: E402,F401  (datos sin dependencias)
MULTIPLE_INSTR = ("Complete the following {lang} function. Return the complete function (signature included, plus any helper "
                  "functions it needs) in a single ```{fence} code block. Do not include a main function or tests.\n\n"
                  "```{fence}\n{prompt}```")


def _entry_name(prompt: str) -> str:
    sig = [l for l in prompt.splitlines() if l.strip()][-1]
    m = re.search(r"(\w+)\s*\(", sig)
    return m.group(1) if m else ""


def multiple_prompt_fn(lang: str):
    lname, fence, _ = MULTIPLE_LANGS[lang]

    def fn(line: dict, task_name: str = f"multiple_{lang}") -> Doc:
        return Doc(task_name=task_name, query=MULTIPLE_INSTR.format(lang=lname, fence=fence, prompt=line["prompt"]),
                   choices=[""], gold_index=0,
                   specific={"prompt": line["prompt"], "tests": line["tests"], "name": line["name"], "language": lang,
                             "entry": _entry_name(line["prompt"])})
    fn.__name__ = f"multiple_{lang}_prompt"
    return fn


def _defines(code: str, name: str, lang: str) -> bool:
    n = re.escape(name)
    if lang == "js":
        return bool(re.search(rf"\bfunction\s*\*?\s+{n}\s*\(|\b(?:const|let|var)\s+{n}\s*=", code))
    if lang == "rs":
        return bool(re.search(rf"\bfn\s+{n}\b", code))
    if lang == "go":
        return bool(re.search(rf"\bfunc\s+{n}\s*\(", code))
    # C++ / Java: «tipo nombre(…) {» al principio de línea (no `return nombre(…)` ni llamadas)
    for m in re.finditer(rf"(?m)^[ \t]*(?!return\b)(?:[\w:<>,\[\]*&?.]+[ \t]+)+[*&]?{n}\s*\(", code):
        rest = code[m.end():]
        b, s = rest.find("{"), rest.find(";")
        if b >= 0 and (s < 0 or b < s):
            return True
    return False


def _body_program(prompt: str, completion: str, tests: str, lang: str) -> str:
    """Modo MultiPL-E clásico: prompt + cuerpo (cortado donde se cierra la función) + tests."""
    cut = _close_index(completion, lang)
    if cut is not None:
        completion = completion[:cut]
    closer = "}\n" if lang in ("js", "go") else ""        # en cpp/java/rs los tests empiezan cerrando la función
    return prompt + completion.rstrip() + "\n" + closer + tests


def _strip_leading_close(tests: str) -> str:
    """Los tests de cpp/java/rs empiezan con la `}` que cierra la función del prompt; en modo «función completa» sobra."""
    return re.sub(r"^\s*\}", "", tests, count=1)


BITS_STDCXX = """#pragma once
#include <algorithm>
#include <any>
#include <array>
#include <bitset>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
"""

JAVATUPLES_PAIR = """package org.javatuples;

import java.util.Objects;

public class Pair<A, B> {
    private final A value0;
    private final B value1;
    public Pair(A value0, B value1) { this.value0 = value0; this.value1 = value1; }
    public static <A, B> Pair<A, B> with(A value0, B value1) { return new Pair<A, B>(value0, value1); }
    public A getValue0() { return value0; }
    public B getValue1() { return value1; }
    public Object getValue(int pos) { if (pos == 0) return value0; if (pos == 1) return value1; throw new IllegalArgumentException(); }
    public int getSize() { return 2; }
    public <X> Pair<X, B> setAt0(X v) { return new Pair<X, B>(v, value1); }
    public <X> Pair<A, X> setAt1(X v) { return new Pair<A, X>(value0, v); }
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Pair)) return false;
        Pair<?, ?> p = (Pair<?, ?>) o;
        return Objects.equals(value0, p.value0) && Objects.equals(value1, p.value1);
    }
    @Override public int hashCode() { return Objects.hash(value0, value1); }
    @Override public String toString() { return "[" + value0 + ", " + value1 + "]"; }
}
"""


def _cpp_program(code: str, spec: dict) -> str:
    prompt, tests = spec["prompt"], spec["tests"]
    if not _defines(code, spec["entry"], "cpp"):
        return _body_program(prompt, code, tests, "cpp")
    code = _remove_blocks(code, r"^[ \t]*(?:int|void|signed|auto)\s+main\s*\(", "cpp")
    includes = "\n".join(l for l in prompt.splitlines() if l.startswith("#include"))
    return includes + "\n" + code.rstrip() + "\n" + _strip_leading_close(tests)


def _js_program(code: str, spec: dict) -> str:
    if not _defines(code, spec["entry"], "js"):
        return _body_program(spec["prompt"], code, spec["tests"], "js")
    code = re.sub(r"(?m)^\s*export\s+(?:default\s+)?", "", code)
    code = re.sub(r"(?m)^\s*module\.exports\b.*$", "", code)
    return code.rstrip() + "\n\n{\n" + spec["tests"] + "\n}\n"           # bloque: el `const assert` de los tests no choca


def _rs_program(code: str, spec: dict) -> str:
    prompt, tests = spec["prompt"], spec["tests"]
    if not _defines(code, spec["entry"], "rs"):
        return _body_program(prompt, code, tests, "rs")
    code = _remove_blocks(code, r"^[ \t]*(?:pub\s+)?fn\s+main\s*\(", "rs")
    code = _remove_blocks(code, r"^[ \t]*#\[cfg\(test\)\]\s*\n[ \t]*mod\s+\w+", "rs")
    uses = "\n".join(l for l in prompt.splitlines() if l.startswith("use ") and l.strip() not in code)
    return uses + "\n" + code.rstrip() + "\n" + _strip_leading_close(tests)


_JAVA_CLASS_RE = re.compile(r"(?m)^[ \t]*(?:(?:public|final|abstract|static)\s+)*(class|interface|record|enum)\s+(\w+)[^{;]*\{")


def _java_program(code: str, spec: dict) -> str:
    prompt, tests = spec["prompt"], spec["tests"]
    if not _defines(code, spec["entry"], "java"):
        return _body_program(prompt, code, tests, "java")
    imports = [l.strip() for l in code.splitlines() if re.match(r"\s*import\s+[\w.*]+\s*;", l)]
    code = re.sub(r"(?m)^\s*(?:import|package)\s+[\w.*\s]+;[ \t]*$", "", code)
    # ¿la función está dentro de una clase (p. ej. `class Problem {…}` o `class Solution {…}`)? → nos quedamos con sus miembros
    members, extra = code, ""
    for m in _JAVA_CLASS_RE.finditer(code):
        ob = m.end() - 1
        depth_before = sum(1 if c == "{" else -1 for _, c in _scan(code[:m.start()], "java"))
        if depth_before != 0:
            continue
        cb = _match_brace(code, ob, "java")
        inner = code[ob + 1:cb]
        if _defines(inner, spec["entry"], "java"):
            members = inner
            outside = code[:m.start()] + code[cb + 1:]
            extra = re.sub(r"(?m)^([ \t]*)public\s+(?=(?:final\s+|abstract\s+)*(?:class|interface|record|enum)\b)", r"\1", outside)
            break
    members = _remove_blocks(members, r"^[ \t]*(?:public\s+)?static\s+void\s+main\s*\(", "java")
    head = prompt[:prompt.index("class Problem {") + len("class Problem {")] if "class Problem {" in prompt else prompt
    head_imports, rest = head.rsplit("\n", 1) if "\n" in head else ("", head)
    extra_imports = "\n".join(i for i in dict.fromkeys(imports) if i not in head_imports)
    return (head_imports + "\n" + extra_imports + "\n" + rest + "\n" + members.rstrip() + "\n"
            + _strip_leading_close(tests) + "\n" + extra.strip() + "\n")


_GO_IMPORT_BLOCK = re.compile(r"(?m)^import\s*\((.*?)\)", re.S)
_GO_IMPORT_LINE = re.compile(r"(?m)^import\s+((?:\w+\s+)?\"[^\"]+\")")


def _go_imports(src: str) -> tuple[list[str], str]:
    specs = []
    for m in _GO_IMPORT_BLOCK.finditer(src):
        specs += [l.strip() for l in m.group(1).splitlines() if l.strip() and not l.strip().startswith("//")]
    src = _GO_IMPORT_BLOCK.sub("", src)
    specs += _GO_IMPORT_LINE.findall(src)
    src = _GO_IMPORT_LINE.sub("", src)
    return specs, src


def _go_program(code: str, spec: dict) -> str:
    prompt, tests = spec["prompt"], spec["tests"]
    if not _defines(code, spec["entry"], "go"):
        return _body_program(prompt, code, tests, "go")
    pkg = re.search(r"(?m)^package\s+\w+", prompt).group(0)
    prompt_imports, _ = _go_imports(prompt)
    model_imports, code = _go_imports(code)
    code = re.sub(r"(?m)^package\s+\w+\s*$", "", code)
    code = _remove_blocks(code, r"^func\s+main\s*\(\s*\)", "go")
    body = code + "\n" + tests
    keep = list(prompt_imports)
    for imp in model_imports:                                  # quitar imports que solo usaba el main eliminado
        m = re.match(r"(?:(\w+)\s+)?\"([^\"]+)\"", imp)
        if not m or imp in keep:
            continue
        alias = m.group(1) or m.group(2).rsplit("/", 1)[-1]
        if alias == "_" or re.search(rf"\b{re.escape(alias)}\.", code):
            keep.append(imp)
    imports = "import (\n" + "".join(f"    {i}\n" for i in dict.fromkeys(keep)) + ")\n"
    return f"{pkg}\n\n{imports}\n{code.strip()}\n\n{tests}"


def run_multiple(code: str, spec: dict) -> tuple[bool, str]:
    """Compiles and runs a MultiPL-E program with its tests in the slmbench sandbox (slmbench.sandbox.execute)."""
    from slmbench.sandbox import execute
    lang = spec["language"]
    if lang == "js":
        return execute({"main.js": _js_program(code, spec)}, [(["node", "main.js"], RUN_TIMEOUT, "run")])
    if lang == "cpp":
        files = {"inc/bits/stdc++.h": BITS_STDCXX, "main.cpp": _cpp_program(code, spec)}   # libc++ (macOS) has no <bits/stdc++.h>
        return execute(files, [(["c++", "-std=c++17", "-w", "-I", "inc", "-o", "prog", "main.cpp"], COMPILE_TIMEOUT, "compile"),
                               (["./prog"], RUN_TIMEOUT, "run")])
    if lang == "java":
        files = {"org/javatuples/Pair.java": JAVATUPLES_PAIR, "Problem.java": _java_program(code, spec)}  # minimal javatuples stand-in
        return execute(files, [(["javac", "-nowarn", "-encoding", "UTF-8", "-d", "out", "Problem.java", "org/javatuples/Pair.java"],
                                COMPILE_TIMEOUT, "compile"),
                               (["java", "-ea", "-Xss16m", "-cp", "out", "Problem"], RUN_TIMEOUT, "run")])
    if lang == "rs":
        return execute({"main.rs": _rs_program(code, spec)},
                       [(["rustc", "--edition", "2021", "-A", "warnings", "-o", "prog", "main.rs"], COMPILE_TIMEOUT, "compile"),
                        (["./prog"], RUN_TIMEOUT, "run")])
    if lang == "go":
        files = {"go.mod": "module slmbench\n\ngo 1.18\n", "main_test.go": _go_program(code, spec)}
        return execute(files, [(["go", "test", "-vet=off", "-count=1", "."], COMPILE_TIMEOUT + RUN_TIMEOUT, "run")])
    return False, f"unknown language {lang}"


# ======================================================================================
# Métricas y tareas
# ======================================================================================
# pass@k: se generan PASSK_N respuestas por problema (SLMBENCH_PASSK_N, por defecto 5) y se estima pass@1 y pass@5 con el
# estimador insesgado de Chen et al. (2021): pass@k = 1 - C(n-c, k) / C(n, k), con c = respuestas correctas.
PASSK_N = max(1, int(os.environ.get("SLMBENCH_PASSK_N") or 5))
PASS_KS = [k for k in (1, 5) if k <= PASSK_N]
PASSK_METRICS = [f"pass@{k}" for k in PASS_KS]


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


class PassAtK(SamplingMetric, SampleLevelComputation):
    """pass@1 y pass@5 a partir de las n respuestas de cada problema. `check(texts, doc)` devuelve cuántas son correctas."""

    def __init__(self, check, n: int = PASSK_N):              # no llama a SamplingMetric.__init__ (solo configura normalizaciones)
        self.check, self.n = check, n

    def num_samples(self) -> int:                              # lighteval genera max(num_samples) respuestas por problema
        return self.n

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        texts = [t or "" for t in list(model_response.final_text or [])[: self.n]] or [""]
        try:
            c = int(self.check(texts, doc))
        except Exception as e:                                 # un fallo del arnés nunca tumba la evaluación
            logger.warning("code scorer error: %r", e)
            c = 0
        return {f"pass@{k}": pass_at_k(len(texts), c, k) for k in PASS_KS}


def passk_metric(check) -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(metric_name=list(PASSK_METRICS), higher_is_better={m: True for m in PASSK_METRICS},
                                     category=SamplingMethod.GENERATIVE, sample_level_fn=PassAtK(check),
                                     corpus_level_fn={m: np.nanmean for m in PASSK_METRICS}, batched_compute=False)


def _count_passing(scorer):
    """Checks the n answers of a problem at the same time (each one in the sandbox)."""
    def one(t, spec):
        try:
            return bool(scorer(t, spec)[0])
        except Exception as e:                                 # noqa: BLE001
            logger.warning("code scorer error: %r", e)
            return False

    def check(texts, doc):
        if len(texts) == 1:
            return int(one(texts[0], doc.specific))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(texts)) as ex:
            return sum(ex.map(lambda t: one(t, doc.specific), texts))
    return check


def _metric(scorer) -> SampleLevelMetricGrouping:
    return passk_metric(_count_passing(scorer))


SCORERS = {
    "humaneval_plus": lambda text, spec: score_humaneval(extract_code(text, "python"), spec),
    "mbpp_plus": lambda text, spec: score_mbpp(extract_code(text, "python"), spec),
    "cruxeval_o": lambda text, spec: score_crux(text, spec, "o"),
    "cruxeval_i": lambda text, spec: score_crux(text, spec, "i"),
}
for _l in MULTIPLE_LANGS:
    SCORERS[f"multiple_{_l}"] = (lambda lang: lambda text, spec: run_multiple(extract_code(text, lang), spec))(_l)


def _task(name: str, prompt_fn, repo: str, subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(name=name, prompt_function=prompt_fn, hf_repo=repo, hf_subset=subset, hf_avail_splits=["test"],
                               evaluation_splits=["test"], few_shots_split=None, few_shots_select=None, generation_size=4096,
                               metrics=[_metric(SCORERS[name])], stop_sequence=[], version=0)


TASKS = [
    _task("humaneval_plus", humaneval_plus_prompt, "evalplus/humanevalplus", "default"),
    _task("mbpp_plus", mbpp_plus_prompt, "evalplus/mbppplus", "default"),
    _task("cruxeval_o", cruxeval_o_prompt, "cruxeval-org/cruxeval", "default"),
    _task("cruxeval_i", cruxeval_i_prompt, "cruxeval-org/cruxeval", "default"),
] + [_task(f"multiple_{l}", multiple_prompt_fn(l), "nuprl/MultiPL-E", f"humaneval-{l}") for l in MULTIPLE_LANGS]
