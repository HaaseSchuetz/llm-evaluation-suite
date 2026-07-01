# LLM Evaluation Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HaaseSchuetz/llm-evaluation-suite/blob/main/notebooks/demo.ipynb)

**A modular, extensible framework for evaluating large language models on standard benchmarks.**

---

## **Why This Project?**
- **Comprehensive**: Supports **MMLU, TruthfulQA, ARC, HellaSwag, BoolQ**, and custom benchmarks.
- **Flexible**: Works with **Hugging Face models** (including quantized/LoRA) and **OpenAI API models**.
- **Rigorous**: Uses the **`lm-evaluation-harness`** (the gold standard for LLM evaluation).
- **Production-Ready**: Generate **reports in Markdown, CSV, or JSON** for sharing.
- **Reproducible**: Full config management and logging.

---

## **Features**
   Feature               | Description                                  |
 |-----------------------|----------------------------------------------|
 | **Multi-Benchmark**   | Evaluate on **5+ standard benchmarks**.       |
 | **Multi-Model**       | Test **Hugging Face + OpenAI API models**.   |
 | **Quantization**      | Supports **4-bit quantized models**.         |
 | **LoRA Support**      | Evaluate **fine-tuned models**.              |
 | **Reporting**         | Generate **Markdown/CSV/JSON reports**.      |
 | **CLI & Python API**  | Use as a **library or command-line tool**.    |

---

## **Setup**
### 1. Clone the Repo
```bash
git clone https://github.com/HaaseSchuetz/llm-evaluation-suite.git
cd llm-evaluation-suite
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional - Set up API Keys
For OpenAI models, set your API Key
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage 
Evaluate a single model
```bash
# Evaluate Mistral-7B on MMLU
python scripts/evaluate.py --model mistral-7b --benchmark mmlu

# With a limit (for testing)
python scripts/evaluate.py --model mistral-7b --benchmark mmlu --limit 100
```

Compare multiple models
```bash
# Compare Mistral-7B and GPT-3.5 on MMLU and TruthfulQA
python scripts/compare.py \
    --models mistral-7b gpt-3.5-turbo \
    --benchmarks mmlu truthfulqa \
    --report-format markdown \
    --report-name mistral_vs_gpt
```

Generate report from raw results
```bash
python scripts/report.py --format markdown --name full_report
```

## Supported Benchmarks 
| Benchmark | Description | Metrics | Few-Shot |
| --- | --- | --- | --- |
| MMLU | Massive Multitask Language Understanding | Accuracy | 5 |
| TruthfulQA | Measures truthfulness of answers | Accuracy | 0 |
| ARC Challenge | Science reasoning questions | Accuracy | 25 |
| HellaSwag | Common-sense natural language inference | Accuracy | 10 |
| BoolQ | Boolean (yes/no) questions | Accuracy | 0 |

To add more, adapt `config/benchmarks.json` or add to `evaluation/benchmarks/`

## Adding a new model 
Edit `config/models.json`
```json
{
  "my-model": {
    "name": "my-org/my-model",
    "type": "huggingface",
    "dtype": "float16",
    "quantization": "4bit",
    "device_map": "auto"
  }
}
```
## Adding a new metric 
1. Create a new file in evaluation/metrics/ (e.g., f1.py).
2. Implement the compute() method.
3. Add it to evaluation/metrics/__init__.py.

## Exemplary results 
| Model | Benchmark | accuracy |
| --- | --- | --- |
| mistral-7b | MMLU | 0.623 |
| mistral-7b | TruthfulQA | 0.456 |
| gpt-3.5-turbo | MMLU | 0.689 |
| gpt-3.5-turbo | TruthfulQA | 0.512 |

## Citing this work 
```bibtex
@misc{llm-evaluation-suite,
  author = {Haase-Schütz, Christian},
  title = {LLM Evaluation Suite},
  year = {2026},
  url = {https://github.com/HaaseSchuetz/llm-evaluation-suite},
  note = {A modular framework for evaluating large language models}
}
```

## Acknowledgments 
* [EleutherAI](https://github.com/EleutherAI) for lm-evaluation-harness.
* [Hugging Face](https://huggingface.co/) for transformers and datasets.
* [Mistral AI](https://mistral.ai/) for open-source LLMs.
* [OpenAI](https://openai.com/) for API access to GPT models.

