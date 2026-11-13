"""Tareas propias de slmbench para lighteval (`--custom-tasks`): benchmarks de Hugging Face que lighteval no trae de serie, y
LiveCodeBench con pass@k."""
from slmbench.benchmarks import ifstruct

TASKS_TABLE = list(ifstruct.TASKS)
try:                                  # tareas de programación (HumanEval+, MBPP+, CRUXEval, MultiPL-E…)
    from slmbench.benchmarks import code as _code
    TASKS_TABLE += list(_code.TASKS)
except ImportError:
    pass
try:                                  # LiveCodeBench (prompt y ejecutor de lighteval, métrica pass@1/pass@5 de slmbench)
    from slmbench.benchmarks import lcb as _lcb
    TASKS_TABLE += list(_lcb.TASKS)
except ImportError:
    pass
try:                                  # MGSM in Spanish, RULER (long context) and BFCL (native tool calling)
    from slmbench.benchmarks import bfcl as _bfcl
    from slmbench.benchmarks import mgsm as _mgsm
    from slmbench.benchmarks import ruler as _ruler
    TASKS_TABLE += list(_mgsm.TASKS) + list(_ruler.TASKS) + list(_bfcl.TASKS)
except ImportError:
    pass
