<!-- register: attribution notice | reader: anyone checking what this repository reuses and under which terms | consumed: looked up by project name -->
# Third-party notices

This repository redistributes no third-party code, model weights, benchmark questions, reference SQL or database files. The projects below are named because the study used them or because a script downloads from them.

## Benchmarks

**BIRD.** Jinyang Li et al., "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs", NeurIPS 2023. <https://bird-bench.github.io/>. Data license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The evaluation used dev release `bird_sql_dev_20251106`. Files under `evidence/` refer to BIRD dev questions only by position (`dev-0` to `dev-1533`) and hold grades, token counts and status fields produced by this study.

**BIRD Mini-Dev scorer.** <https://github.com/bird-bench/mini_dev>, which states CC BY-SA 4.0 in its README. `scripts/fetch_bird_scorer.py` downloads `evaluation/evaluation_ex.py` and `evaluation/evaluation_utils.py` unmodified from commit `f9d2750c9b53639820c9d47f6d7a5e5025c780ab`, verifies their SHA-256 digests, and writes them to a directory that git ignores. Those two files remain under their upstream terms.

**Spider.** Tao Yu et al., "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task", EMNLP 2018. <https://yale-lily.github.io/spider>. Data license: CC BY-SA 4.0. Files under `evidence/spider/` refer to Spider dev questions only by position (`spider-dev-0000` to `spider-dev-1033`).

## Software and model

| Project | Source | License |
| --- | --- | --- |
| Miles | <https://github.com/radixark/miles> | Apache-2.0 |
| Megatron-LM (Miles fork) | <https://github.com/radixark/Megatron-LM> | BSD-style and Apache-2.0, as stated in its `LICENSE` |
| SGLang | <https://github.com/sgl-project/sglang> | Apache-2.0 |
| Qwen3.5-9B | <https://huggingface.co/Qwen/Qwen3.5-9B> | Apache-2.0 |
| func_timeout | <https://pypi.org/project/func-timeout/> | LGPLv2, as stated on PyPI |

Code under `native/` imports Miles and SGLang at run time. `func_timeout` is installed by the user and is needed only for execution scoring.

## Adapters

The LoRA adapters trained in this study are not part of this repository and are not covered by its license.
