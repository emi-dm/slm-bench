# Benchmarks used

slmbench only uses **existing public benchmarks**, run with [lighteval](https://github.com/huggingface/lighteval) against your server. The ones that ship with lighteval are used as they are; the rest have been adapted as custom tasks (see [adapting-a-benchmark.md](adapting-a-benchmark.md)). This document is generated with `slmbench benchmarks --markdown > docs/benchmarks.md`.

## Mode `general` — General

Instructions, maths (also in Spanish) and structured output.

| Benchmark | What it measures | Dataset | Tasks | Metric | Origin |
|---|---|---|--:|---|---|
| [IFEval](#ifeval) | Verifiable instruction following (length, format, required words…). Measures whether the model obeys, not what it knows. | `google/IFEval` | 541 | `prompt_level_strict_acc` | lighteval |
| [GSM8K](#gsm8k) | Grade-school math word problems: multi-step arithmetic reasoning within reach of a small model. | `openai/gsm8k:main` | 1,319 | `extractive_match` | lighteval |
| [MATH-500](#math_500) | 500 competition math problems (MATH subset). Harder: separates small models that reason well. | `HuggingFaceH4/MATH-500` | 500 | `pass@k:k=1&n=1` | lighteval |
| [IFStruct](#ifstruct) | Structured output (Liquid AI): produce JSON or YAML that follows a schema and every requested constraint (item count, wrapper key, code fence, no commentary, enums, ranges). A response scores only if everything holds. | `LiquidAI/ifstruct-v1.0:default/test` | 2,000 | `pass_rate` | custom task |
| [MGSM (es)](#mgsm_es) | Grade-school maths problems human-translated into Spanish (MGSM). Measures reasoning in Spanish; scores the last number of the answer, like the official evaluation. | `juletxara/mgsm:es/test` | 250 | `accuracy` | custom task |

## Mode `code` — Coding

Coding only: Python and other languages, code generation and code reasoning.

| Benchmark | What it measures | Dataset | Tasks | Metric | Origin |
|---|---|---|--:|---|---|
| [HumanEval+](#humaneval_plus) | Complete 164 Python functions from their docstrings, checked with EvalPlus's extended tests (~80x HumanEval's); short self-contained problems that suit small models. | `evalplus/humanevalplus` | 164 | `pass@1` | custom task |
| [MBPP+](#mbpp_plus) | Write Python functions for 378 entry-level tasks described in one sentence (cleaned MBPP) with EvalPlus's extended tests; measures simple everyday coding within reach of small models. | `evalplus/mbppplus` | 378 | `pass@1` | custom task |
| [LiveCodeBench](#lcb) | Recent LeetCode/AtCoder/Codeforces problems (contamination-free). Very hard for SLMs; ≈1.6 GB download. | `lighteval/code_generation_lite:v4_v5` | 268 | `pass@1` | custom task |
| [CRUXEval-O](#cruxeval_o) | Predict the output of 800 short Python functions for a given input; measures code understanding and mental execution without writing long programs, a good fit for small models. | `cruxeval-org/cruxeval` | 800 | `pass@1` | custom task |
| [CRUXEval-I](#cruxeval_i) | Find an input that makes a short Python function return a given output (800 cases, verified by execution); measures backward reasoning about code with short answers. | `cruxeval-org/cruxeval` | 800 | `pass@1` | custom task |
| [MultiPL-E C++](#multiple_cpp) | HumanEval translated to C++ (161 problems), compiled and run against its tests; measures whether a small model can code beyond Python. | `nuprl/MultiPL-E:humaneval-cpp` | 161 | `pass@1` | custom task |
| [MultiPL-E Go](#multiple_go) | HumanEval translated to Go (154 problems), compiled and run against its tests; measures whether a small model can code beyond Python. | `nuprl/MultiPL-E:humaneval-go` | 154 | `pass@1` | custom task |
| [MultiPL-E Java](#multiple_java) | HumanEval translated to Java (158 problems), compiled and run against its tests; measures whether a small model can code beyond Python. | `nuprl/MultiPL-E:humaneval-java` | 158 | `pass@1` | custom task |
| [MultiPL-E JavaScript](#multiple_js) | HumanEval translated to JavaScript (161 problems), compiled and run against its tests; measures whether a small model can code beyond Python. | `nuprl/MultiPL-E:humaneval-js` | 161 | `pass@1` | custom task |
| [MultiPL-E Rust](#multiple_rs) | HumanEval translated to Rust (156 problems), compiled and run against its tests; measures whether a small model can code beyond Python. | `nuprl/MultiPL-E:humaneval-rs` | 156 | `pass@1` | custom task |

## Mode `tools` — Tool calling

Native function calling (BFCL): picking the right function and arguments.

| Benchmark | What it measures | Dataset | Tasks | Metric | Origin |
|---|---|---|--:|---|---|
| [BFCL Simple](#bfcl_simple) | Berkeley Function Calling Leaderboard, simple category: one function, one call. Native tool calling (tools / tool_calls), judged with BFCL's AST checker. | `bfcl:simple` | 400 | `pass@1` | custom task |
| [BFCL Multiple](#bfcl_multiple) | Berkeley Function Calling Leaderboard, multiple category: pick the right function out of 2–4. Native tool calling (tools / tool_calls), judged with BFCL's AST checker. | `bfcl:multiple` | 200 | `pass@1` | custom task |
| [BFCL Parallel](#bfcl_parallel) | Berkeley Function Calling Leaderboard, parallel category: several calls to the same function in one answer. Native tool calling (tools / tool_calls), judged with BFCL's AST checker. | `bfcl:parallel` | 200 | `pass@1` | custom task |
| [BFCL Parallel multiple](#bfcl_parallel_multiple) | Berkeley Function Calling Leaderboard, parallel multiple category: several calls to several functions. Native tool calling (tools / tool_calls), judged with BFCL's AST checker. | `bfcl:parallel_multiple` | 200 | `pass@1` | custom task |

## Mode `long` — Long context

Retrieving information from long contexts (RULER, 4k and 8k tokens).

| Benchmark | What it measures | Dataset | Tasks | Metric | Origin |
|---|---|---|--:|---|---|
| [RULER NIAH 4k](#ruler_4k) | Needle in a haystack (RULER) with ~4k tokens of context: find one or several values hidden in a long text. Measures whether the model really uses its context. | `simonjegou/ruler:4096/test` | 4,000 | `string_match_all` | custom task |
| [RULER NIAH 8k](#ruler_8k) | Needle in a haystack (RULER) with ~8k tokens of context: find one or several values hidden in a long text. Measures whether the model really uses its context. | `simonjegou/ruler:8192/test` | 4,000 | `string_match_all` | custom task |

## Validation and quality notes

Before using any model, every adapted benchmark was validated by scoring its **reference answers** (they should score ~100%). All the code checks run inside the slmbench sandbox (a Docker Sandboxes microVM with no network and no access to your files) and were re-validated there: the same pass counts as below, and 22/22 MultiPL-E hand-written controls.

| Benchmark | Reference answers that pass | Note |
|---|--:|---|
| HumanEval+ | 163/164 | The HF test for HumanEval/32 (`find_zero`) calls `_poly` with its arguments swapped; slmbench fixes that test. Even so, the reference solution fails 2 of 886 ill-conditioned inputs |
| MBPP+ | 377/378 | Task 590 differs only in the last decimal digit (precision of the stored value) |
| CRUXEval-O / CRUXEval-I | 800/800 · 800/800 | — |
| MultiPL-E | — | The dataset has no reference solutions: it was validated with hand-written solutions (correct and incorrect) in all 5 languages and by compiling all 791 problems. Three do not compile because of defects in the dataset, so nobody can pass them: JS HumanEval_51, C++ HumanEval_137 and Go HumanEval_87 |
| IFStruct | — (no reference answers) | Scored with the official validator from `Liquid4All/ifstruct`, vendored unchanged (its 20 upstream tests pass). The prompt is sent alone as the user message, like the official client, and the 2,000 HF rows match the official `data/test.jsonl` exactly (prompts and specs). End to end, on 100 prompts against the same llama-server, slmbench and a client sending exactly the official request gave the **same pass rate (11.0% vs 11.0%)**, identical answers on 89/100 and the same pass/fail on 96/100 (2 differences each way). Those differences come from llama.cpp's nondeterminism under concurrent load: the same client flips pass/fail between calls on 2 of them, and sent alone slmbench's exact request reproduces the official-style answer |
| LiveCodeBench | correct / wrong / timeout controls behave as expected | lighteval's prompt, dataset and tests (`run_test`); slmbench adds pass@1/pass@5 over n answers. **lighteval's own LCB checker scores every answer 0 on macOS** (it starts a process with a nested function, which the default `spawn` start method can't pickle), even a correct one: slmbench uses a fixed copy with the same per-test results as lighteval's under `fork` (verified on a correct, a wrong and a timing-out solution to abc366_a, all 21 tests). Answers get at least 8,192 tokens (with 2,048 every answer was truncated); small reasoning models benefit from `--max-tokens 16384` |
| BFCL (4 categories) | 1000/1000 · broken answers rejected 1000/1000 | AST checker ported from BFCL (eval_checker/ast_eval/ast_checker.py): calls built from every official answer, with parallel calls shuffled, all pass; changing one argument value makes every one fail. Two quirks of the HF v3 data are handled like the official checker: simple_363 names the function differently in questions and answers, and some answers allow optional parameters that aren't in the schema |
| MGSM (es) | 250/250 · wrong number accepted 0/250 | Official criterion (last number of the answer), also with thousands separators |
| RULER NIAH 4k / 8k | 4000/4000 · 4000/4000 | string_match_all as in RULER; an answer missing one value always scores below 100% |

pass@k (coding): n answers per problem (default 5, sampled at temperature 0.6) and the unbiased estimator of Chen et al. (2021), identical to lighteval's `estimate_pass_at_k`. With 5 copies of the reference solution a problem scores pass@1 = pass@5 = 1; with 2 correct out of 5, pass@1 = 0.4 and pass@5 = 1.

The lighteval benchmarks (IFEval, GSM8K, MATH-500) use lighteval's implementation unchanged.

## Benchmark cards

<a id="ifeval"></a>

### IFEval  ·  `ifeval`

- **Mode**: general  ·  **Metric**: `prompt_level_strict_acc`  ·  **Tasks**: 541
- **Dataset**: `google/IFEval`  ·  **lighteval**: `ifeval|0`
- **Reference**: https://arxiv.org/abs/2311.07911

Verifiable instruction following (length, format, required words…). Measures whether the model obeys, not what it knows.

## Sample task (what the model receives)

**user**

````text
Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.
````


---

<a id="gsm8k"></a>

### GSM8K  ·  `gsm8k`

- **Mode**: general  ·  **Metric**: `extractive_match`  ·  **Tasks**: 1,319
- **Dataset**: `openai/gsm8k:main`  ·  **lighteval**: `gsm8k|0`
- **Reference**: https://arxiv.org/abs/2110.14168

Grade-school math word problems: multi-step arithmetic reasoning within reach of a small model.

## Sample task (what the model receives)

**user**

````text
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer:
````

**Reference answer**

````text
 Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18
````


---

<a id="math_500"></a>

### MATH-500  ·  `math_500`

- **Mode**: general  ·  **Metric**: `pass@k:k=1&n=1`  ·  **Tasks**: 500
- **Dataset**: `HuggingFaceH4/MATH-500`  ·  **lighteval**: `math_500|0`
- **Reference**: https://arxiv.org/abs/2305.20050

500 competition math problems (MATH subset). Harder: separates small models that reason well.

## Sample task (what the model receives)

**user**

````text
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
````

**Reference answer**

````text
ANSWER: We have that $r = \sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\frac{\pi}{2}$ with the positive $x$-axis.

[asy]
unitsize(0.8 cm);

draw((-0.5,0)--(3.5,0));
draw((0,-0.5)--(0,3.5));
draw(arc((0,0),3,0,90),red,Arrow(6));

dot((0,3), red);
label("$(0,3)$", (0,3), W);
dot((3,0), red);
[/asy]

Therefore, the polar coordinates are $\boxed{\left( 3, \frac{\pi}{2} \right)}.$
````


---

<a id="ifstruct"></a>

### IFStruct  ·  `ifstruct`

- **Mode**: general  ·  **Metric**: `pass_rate`  ·  **Tasks**: 2,000
- **Dataset**: `LiquidAI/ifstruct-v1.0:default/test`  ·  **lighteval**: `ifstruct|0` (slmbench custom task)
- **Reference**: https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0

Structured output (Liquid AI): produce JSON or YAML that follows a schema and every requested constraint (item count, wrapper key, code fence, no commentary, enums, ranges). A response scores only if everything holds.

## Sample task (what the model receives)

**user**

````text
need some gothic suspense story chapters generated, dialogue-heavy. let me spec this out.

format is yaml, in a fenced code block. root should be a bare array — actually wait, do i want it keyed under something like `chapters:`? no. keep it a bare array, not wrapped in an object. say a comment or two alongside it if you want, i'm not policing commentary.

count: give me 3-4 chapters. between 3 and 4 inclusive.

each chapter is an object. fields: `chapter_title` (string, short title or editorial label), then `metadata` (an object, detail below), then `chapter_text` — a string, multiline, actual polished prose with dialogue-heavy passages, not outline notes. work one recalled line of dialogue into the narration where it reads natural.

now `metadata`, the nested object. it has:
- `metadata.p
… (2,203 chars) …
eep it a boolean, true/false.

hold on, do i also want a `word_count` field on each chapter? skip it. and don't add a chapter number field either — no extra keys anywhere, at any level. exactly the fields i listed: top level `chapter_title`, `metadata`, `chapter_text`, and inside metadata the six i named.

one more pass on `section_role` — leave it the six values as stated, that's settled. metadata should read like real manuscript context, not notes about the schema.

so: yaml, fenced code block, bare list at root, 3 to 4 items.
````


---

<a id="mgsm_es"></a>

### MGSM (es)  ·  `mgsm_es`

- **Mode**: general  ·  **Metric**: `accuracy`  ·  **Tasks**: 250
- **Dataset**: `juletxara/mgsm:es/test`  ·  **lighteval**: `mgsm_es|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2210.03057

Grade-school maths problems human-translated into Spanish (MGSM). Measures reasoning in Spanish; scores the last number of the answer, like the official evaluation.

## Sample task (what the model receives)

**user**

````text
Resuelve el siguiente problema razonando paso a paso. Termina tu respuesta con una línea «La respuesta es N.», donde N es el resultado numérico.

Pregunta: Los patos de Janet ponen 16 huevos por día. Ella come tres en el desayuno todas las mañanas y usa cuatro para hornear magdalenas para sus amigos todos los días. Vende lo que sobra en el mercado de productores diariamente a $2 el huevo fresco de pato. ¿Cuánto gana en dólares todos los días en el mercado de productores?
Respuesta paso a paso:
````

**Reference answer**

````text
18
````


---

<a id="bfcl_simple"></a>

### BFCL Simple  ·  `bfcl_simple`

- **Mode**: tools  ·  **Metric**: `pass@1`  ·  **Tasks**: 400
- **Dataset**: `bfcl:simple`  ·  **lighteval**: `bfcl_simple|0` (slmbench custom task)
- **Reference**: https://gorilla.cs.berkeley.edu/leaderboard.html

Berkeley Function Calling Leaderboard, simple category: one function, one call. Native tool calling (tools / tool_calls), judged with BFCL's AST checker.

## Sample task (what the model receives)

**user**

````text
Find the area of a triangle with a base of 10 units and height of 5 units.
````

**Reference answer**

````text
[{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
````


---

<a id="bfcl_multiple"></a>

### BFCL Multiple  ·  `bfcl_multiple`

- **Mode**: tools  ·  **Metric**: `pass@1`  ·  **Tasks**: 200
- **Dataset**: `bfcl:multiple`  ·  **lighteval**: `bfcl_multiple|0` (slmbench custom task)
- **Reference**: https://gorilla.cs.berkeley.edu/leaderboard.html

Berkeley Function Calling Leaderboard, multiple category: pick the right function out of 2–4. Native tool calling (tools / tool_calls), judged with BFCL's AST checker.

## Sample task (what the model receives)

**user**

````text
Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?
````

**Reference answer**

````text
[{"triangle_properties.get": {"side1": [5], "side2": [4], "side3": [3], "get_area": ["", true], "get_perimeter": ["", true], "get_angles": ["", true]}}]
````


---

<a id="bfcl_parallel"></a>

### BFCL Parallel  ·  `bfcl_parallel`

- **Mode**: tools  ·  **Metric**: `pass@1`  ·  **Tasks**: 200
- **Dataset**: `bfcl:parallel`  ·  **lighteval**: `bfcl_parallel|0` (slmbench custom task)
- **Reference**: https://gorilla.cs.berkeley.edu/leaderboard.html

Berkeley Function Calling Leaderboard, parallel category: several calls to the same function in one answer. Native tool calling (tools / tool_calls), judged with BFCL's AST checker.

## Sample task (what the model receives)

**user**

````text
Play songs from the artists Taylor Swift and Maroon 5, with a play time of 20 minutes and 15 minutes respectively, on Spotify.
````

**Reference answer**

````text
[{"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}}, {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}}]
````


---

<a id="bfcl_parallel_multiple"></a>

### BFCL Parallel multiple  ·  `bfcl_parallel_multiple`

- **Mode**: tools  ·  **Metric**: `pass@1`  ·  **Tasks**: 200
- **Dataset**: `bfcl:parallel_multiple`  ·  **lighteval**: `bfcl_parallel_multiple|0` (slmbench custom task)
- **Reference**: https://gorilla.cs.berkeley.edu/leaderboard.html

Berkeley Function Calling Leaderboard, parallel multiple category: several calls to several functions. Native tool calling (tools / tool_calls), judged with BFCL's AST checker.

## Sample task (what the model receives)

**user**

````text
Find the sum of all the multiples of 3 and 5 between 1 and 1000. Also find the product of the first five prime numbers.
````

**Reference answer**

````text
[{"math_toolkit.sum_of_multiples": {"lower_limit": [1], "upper_limit": [1000], "multiples": [[3, 5]]}}, {"math_toolkit.product_of_primes": {"count": [5]}}]
````


---

<a id="ruler_4k"></a>

### RULER NIAH 4k  ·  `ruler_4k`

- **Mode**: long  ·  **Metric**: `string_match_all`  ·  **Tasks**: 4,000
- **Dataset**: `simonjegou/ruler:4096/test`  ·  **lighteval**: `ruler_niah_4k|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2404.06654

Needle in a haystack (RULER) with ~4k tokens of context: find one or several values hidden in a long text. Measures whether the model really uses its context.

## Sample task (what the model receives)

**user**

````text
A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. Th
… (13,904 chars) …
. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.

What is the special magic number for wandering-age mentioned in the provided text?
Answer in one short sentence that starts with: "The special magic number for wandering-age mentioned in the provided text is"
````

**Reference answer**

````text
8090293
````


---

<a id="ruler_8k"></a>

### RULER NIAH 8k  ·  `ruler_8k`

- **Mode**: long  ·  **Metric**: `string_match_all`  ·  **Tasks**: 4,000
- **Dataset**: `simonjegou/ruler:8192/test`  ·  **lighteval**: `ruler_niah_8k|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2404.06654

Needle in a haystack (RULER) with ~8k tokens of context: find one or several values hidden in a long text. Measures whether the model really uses its context.

## Sample task (what the model receives)

**user**

````text
A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. Th
… (29,648 chars) …
blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.

What is the special magic number for watery-walk mentioned in the provided text?
Answer in one short sentence that starts with: "The special magic number for watery-walk mentioned in the provided text is"
````

**Reference answer**

````text
2166941
````


---

<a id="humaneval_plus"></a>

### HumanEval+  ·  `humaneval_plus`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 164
- **Dataset**: `evalplus/humanevalplus`  ·  **lighteval**: `humaneval_plus|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2305.01210

Complete 164 Python functions from their docstrings, checked with EvalPlus's extended tests (~80x HumanEval's); short self-contained problems that suit small models.

## Sample task (what the model receives)

**user**

````text
Complete the following Python function. Return the complete function (signature included, plus any imports or helper functions it needs) in a single ```python code block.

```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```
````


---

<a id="mbpp_plus"></a>

### MBPP+  ·  `mbpp_plus`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 378
- **Dataset**: `evalplus/mbppplus`  ·  **lighteval**: `mbpp_plus|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2305.01210

Write Python functions for 378 entry-level tasks described in one sentence (cleaned MBPP) with EvalPlus's extended tests; measures simple everyday coding within reach of small models.

## Sample task (what the model receives)

**user**

````text
Write a function to find the shared elements from the given two lists.
Your code should pass this test:

assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))

Write the Python function and return it in a single ```python code block.
````


---

<a id="lcb"></a>

### LiveCodeBench  ·  `lcb`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 268
- **Dataset**: `lighteval/code_generation_lite:v4_v5`  ·  **lighteval**: `livecodebench|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2403.07974

Recent LeetCode/AtCoder/Codeforces problems (contamination-free). Very hard for SLMs; ≈1.6 GB download.

## Sample task (what the model receives)

**user**

````text
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: You are given N linear functions f_1, f_2, \ldots, f_N, where f_i(x) = A_i x + B_i.
Find the maximum possible value of f_{p_1}(f_{p_2}(\ldots f_{p_K}(1) \ldots )) for a sequence p = (p_1, p_2, \ldots, p_K) of K distinct integers between 1 and N, inclusive.

Input

The input is given from Standard Input in the following format:
N K
A_1 B_1
A_2 B_2
\vdots
A_N B_N

Output

Print the answer as an integer.

Constraints


- 1 \leq N \leq 2 \times 10^{5}
- 1 \leq K \leq \text{min}(N,10)
- 1 \leq A_i, B_i \leq 50 (1 \leq i \leq N)
- All input values are integers.

Sample Input 1

3 2
2 3
1 5
4 2

Sample Output 1

26

Here are all possible p and the corresponding values of f_{p_1}(f_{p_2}(1)):

- p= ( 1,2 ) : f_1(f_2(1))=15
- p= ( 1,3 ) : f_1(f_3(1))=15
- p= ( 2,1 ) : f_2(f_1(1))=10
- p= ( 2,3 ) : f_2(f_3(1))=11
- p= ( 3,1 ) : f_3(f_1(1))=22
- p= ( 3,2 ) : f_3(f_2(1))=26

Therefore, print 26.

Sample Input 2

10 3
48 40
34 22
24 37
45 40
48 31
49 44
45 40
44 6
35 22
39 28

Sample Output 2

216223

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```


````


---

<a id="cruxeval_o"></a>

### CRUXEval-O  ·  `cruxeval_o`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 800
- **Dataset**: `cruxeval-org/cruxeval`  ·  **lighteval**: `cruxeval_o|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2401.03065

Predict the output of 800 short Python functions for a given input; measures code understanding and mental execution without writing long programs, a good fit for small models.

## Sample task (what the model receives)

**user**

````text
You are given a Python function and an assertion containing an input to the function. Find the output of executing the function on the given input, even if the function is incorrect or incomplete.

```python
def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output
assert f([1, 1, 3, 1, 3, 1]) == ??
```

You may reason step by step. Then give the output as a Python literal (no expressions or function calls) between [ANSWER] and [/ANSWER], for example: [ANSWER]'abc'[/ANSWER]
````

**Reference answer**

````text
[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]
````


---

<a id="cruxeval_i"></a>

### CRUXEval-I  ·  `cruxeval_i`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 800
- **Dataset**: `cruxeval-org/cruxeval`  ·  **lighteval**: `cruxeval_i|0` (slmbench custom task)
- **Reference**: https://arxiv.org/abs/2401.03065

Find an input that makes a short Python function return a given output (800 cases, verified by execution); measures backward reasoning about code with short answers.

## Sample task (what the model receives)

**user**

````text
You are given a Python function and an output. Find any input such that executing the function on that input leads to the given output. There may be several answers; give only one.

```python
def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output
assert f(??) == [(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]
```

You may reason step by step. Then give the completed assertion between [ANSWER] and [/ANSWER], for example: [ANSWER]assert f(1, 'a') == [(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)][/ANSWER]
````

**Reference answer**

````text
[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]
````


---

<a id="multiple_cpp"></a>

### MultiPL-E C++  ·  `multiple_cpp`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 161
- **Dataset**: `nuprl/MultiPL-E:humaneval-cpp`  ·  **lighteval**: `multiple_cpp|0` (slmbench custom task)
- **Reference**: https://huggingface.co/datasets/nuprl/MultiPL-E
- **Requires**: c++  ✔

HumanEval translated to C++ (161 problems), compiled and run against its tests; measures whether a small model can code beyond Python.

## Sample task (what the model receives)

**user**

````text
Complete the following C++ function. Return the complete function (signature included, plus any helper functions it needs) in a single ```cpp code block. Do not include a main function or tests.

```cpp
#include<assert.h>
#include<bits/stdc++.h>
// Check if in given vector of numbers, are any two numbers closer to each other than
// given threshold.
// >>> has_close_elements((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.0f})), (0.5f))
// (false)
// >>> has_close_elements((std::vector<float>({(float)1.0f, (float)2.8f, (float)3.0f, (float)4.0f, (float)5.0f, (float)2.0f})), (0.3f))
// (true)
bool has_close_elements(std::vector<float> numbers, float threshold) {
```
````


---

<a id="multiple_go"></a>

### MultiPL-E Go  ·  `multiple_go`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 154
- **Dataset**: `nuprl/MultiPL-E:humaneval-go`  ·  **lighteval**: `multiple_go|0` (slmbench custom task)
- **Reference**: https://huggingface.co/datasets/nuprl/MultiPL-E
- **Requires**: go  ✔

HumanEval translated to Go (154 problems), compiled and run against its tests; measures whether a small model can code beyond Python.

## Sample task (what the model receives)

**user**

````text
Complete the following Go function. Return the complete function (signature included, plus any helper functions it needs) in a single ```go code block. Do not include a main function or tests.

```go
package has_close_elements_test

import (
    "testing"
    "fmt"
)

// Check if in given list of numbers, are any two numbers closer to each other than
// given threshold.
// >>> has_close_elements([]float64{1.0, 2.0, 3.0}, 0.5)
// false
// >>> has_close_elements([]float64{1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)
// true
func has_close_elements(numbers []float64, threshold float64) bool {
```
````


---

<a id="multiple_java"></a>

### MultiPL-E Java  ·  `multiple_java`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 158
- **Dataset**: `nuprl/MultiPL-E:humaneval-java`  ·  **lighteval**: `multiple_java|0` (slmbench custom task)
- **Reference**: https://huggingface.co/datasets/nuprl/MultiPL-E
- **Requires**: javac, java  ✔

HumanEval translated to Java (158 problems), compiled and run against its tests; measures whether a small model can code beyond Python.

## Sample task (what the model receives)

**user**

````text
Complete the following Java function. Return the complete function (signature included, plus any helper functions it needs) in a single ```java code block. Do not include a main function or tests.

```java
import java.util.*;
import java.lang.reflect.*;
import org.javatuples.*;
import java.security.*;
import java.math.*;
import java.io.*;
import java.util.stream.*;
class Problem {
    // Check if in given array list of numbers, are any two numbers closer to each other than
    // given threshold.
    // >>> hasCloseElements((new ArrayList<Float>(Arrays.asList((float)1.0f, (float)2.0f, (float)3.0f))), (0.5f))
    // (false)
    // >>> hasCloseElements((new ArrayList<Float>(Arrays.asList((float)1.0f, (float)2.8f, (float)3.0f, (float)4.0f, (float)5.0f, (float)2.0f))), (0.3f))
    // (true)
    public static boolean hasCloseElements(ArrayList<Float> numbers, float threshold) {
```
````


---

<a id="multiple_js"></a>

### MultiPL-E JavaScript  ·  `multiple_js`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 161
- **Dataset**: `nuprl/MultiPL-E:humaneval-js`  ·  **lighteval**: `multiple_js|0` (slmbench custom task)
- **Reference**: https://huggingface.co/datasets/nuprl/MultiPL-E
- **Requires**: node  ✔

HumanEval translated to JavaScript (161 problems), compiled and run against its tests; measures whether a small model can code beyond Python.

## Sample task (what the model receives)

**user**

````text
Complete the following JavaScript function. Return the complete function (signature included, plus any helper functions it needs) in a single ```javascript code block. Do not include a main function or tests.

```javascript
//Check if in given array of numbers, are any two numbers closer to each other than
// given threshold.
// >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
// false
// >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
// true
function has_close_elements(numbers, threshold){
```
````


---

<a id="multiple_rs"></a>

### MultiPL-E Rust  ·  `multiple_rs`

- **Mode**: code  ·  **Metric**: `pass@1`  ·  **Tasks**: 156
- **Dataset**: `nuprl/MultiPL-E:humaneval-rs`  ·  **lighteval**: `multiple_rs|0` (slmbench custom task)
- **Reference**: https://huggingface.co/datasets/nuprl/MultiPL-E
- **Requires**: rustc  ✔

HumanEval translated to Rust (156 problems), compiled and run against its tests; measures whether a small model can code beyond Python.

## Sample task (what the model receives)

**user**

````text
Complete the following Rust function. Return the complete function (signature included, plus any helper functions it needs) in a single ```rust code block. Do not include a main function or tests.

```rust
/// Check if in given vector of numbers, are any two numbers closer to each other than
/// given threshold.
/// >>> has_close_elements(vec![1.0, 2.0, 3.0], 0.5)
/// false
/// >>> has_close_elements(vec![1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
/// true
fn has_close_elements(numbers: Vec<f64>, threshold: f64) -> bool {
```
````


---

