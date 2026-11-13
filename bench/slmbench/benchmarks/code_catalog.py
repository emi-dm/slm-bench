"""Catálogo de los benchmarks de programación (solo datos, sin importar lighteval).

Se separa de `code.py` para que la TUI y el CLI puedan listar benchmarks, leaderboard, etc. sin cargar lighteval
(que arrastra torch y scikit-learn y tarda varios segundos en importarse).
"""
from __future__ import annotations

MULTIPLE_LANGS = {
    # clave: (nombre en el prompt, nombre en la valla, display)
    "js": ("JavaScript", "javascript", "JavaScript"),
    "cpp": ("C++", "cpp", "C++"),
    "java": ("Java", "java", "Java"),
    "rs": ("Rust", "rust", "Rust"),
    "go": ("Go", "go", "Go"),
}

_MULTIPLE_N = {"js": 161, "cpp": 161, "java": 158, "rs": 156, "go": 154}
_MULTIPLE_META = {"js": ("javascript", ["node"]), "cpp": ("cpp", ["c++"]), "java": ("java", ["javac", "java"]),
                  "rs": ("rust", ["rustc"]), "go": ("go", ["go"])}

CATALOG: dict[str, dict] = {
    "humaneval_plus": {
        "name": "HumanEval+", "spec": "humaneval_plus|0", "hf": "evalplus/humanevalplus", "n": 164, "metric": "pass@1", "passk": True,
        "language": "python", "requires": [], "link": "https://arxiv.org/abs/2305.01210",
        "desc_es": "Completar 164 funciones Python a partir de su docstring, verificadas con los tests ampliados de EvalPlus (≈80× más "
                   "casos que HumanEval); problemas cortos y autocontenidos, ideales para medir la programación básica de un modelo pequeño.",
        "desc_en": "Complete 164 Python functions from their docstrings, checked with EvalPlus's extended tests (~80x HumanEval's); "
                   "short self-contained problems that suit small models.",
    },
    "mbpp_plus": {
        "name": "MBPP+", "spec": "mbpp_plus|0", "hf": "evalplus/mbppplus", "n": 378, "metric": "pass@1", "passk": True,
        "language": "python", "requires": [], "link": "https://arxiv.org/abs/2305.01210",
        "desc_es": "Escribir funciones Python para 378 tareas de nivel básico descritas en una frase (MBPP depurado), con tests ampliados "
                   "de EvalPlus; mide programación cotidiana sencilla, al alcance de modelos pequeños.",
        "desc_en": "Write Python functions for 378 entry-level tasks described in one sentence (cleaned MBPP) with EvalPlus's extended "
                   "tests; measures simple everyday coding within reach of small models.",
    },
    "cruxeval_o": {
        "name": "CRUXEval-O", "spec": "cruxeval_o|0", "hf": "cruxeval-org/cruxeval", "n": 800, "metric": "pass@1", "passk": True,
        "language": "python", "requires": [], "link": "https://arxiv.org/abs/2401.03065",
        "desc_es": "Predecir la salida de 800 funciones Python cortas para una entrada dada; mide la comprensión y ejecución mental del "
                   "código sin necesidad de escribir programas largos, adecuado para modelos pequeños.",
        "desc_en": "Predict the output of 800 short Python functions for a given input; measures code understanding and mental "
                   "execution without writing long programs, a good fit for small models.",
    },
    "cruxeval_i": {
        "name": "CRUXEval-I", "spec": "cruxeval_i|0", "hf": "cruxeval-org/cruxeval", "n": 800, "metric": "pass@1", "passk": True,
        "language": "python", "requires": [], "link": "https://arxiv.org/abs/2401.03065",
        "desc_es": "Encontrar una entrada que haga que una función Python corta devuelva una salida dada (800 casos, se verifica "
                   "ejecutando); mide razonamiento inverso sobre código con respuestas breves.",
        "desc_en": "Find an input that makes a short Python function return a given output (800 cases, verified by execution); "
                   "measures backward reasoning about code with short answers.",
    },
}
for _l, (_lname, _fence, _disp) in MULTIPLE_LANGS.items():
    CATALOG[f"multiple_{_l}"] = {
        "name": f"MultiPL-E {_disp}", "spec": f"multiple_{_l}|0", "hf": f"nuprl/MultiPL-E:humaneval-{_l}", "n": _MULTIPLE_N[_l],
        "metric": "pass@1", "passk": True, "language": _MULTIPLE_META[_l][0], "requires": _MULTIPLE_META[_l][1],
        "link": "https://huggingface.co/datasets/nuprl/MultiPL-E",
        "desc_es": f"HumanEval traducido a {_disp} ({_MULTIPLE_N[_l]} problemas), compilado y ejecutado con sus tests; mide si un "
                   f"modelo pequeño programa también fuera de Python.",
        "desc_en": f"HumanEval translated to {_disp} ({_MULTIPLE_N[_l]} problems), compiled and run against its tests; measures "
                   f"whether a small model can code beyond Python.",
    }
