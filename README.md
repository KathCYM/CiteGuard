# CiteGuard

Large Language Models (LLMs) have emerged as powerful assistants for scientific writing. However, concerns remain about the quality and reliability of the generated text, including citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which assesses whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves over the prior baseline by 10 percentage points and achieves up to 68.1% accuracy on the CiteME benchmark, approaching human performance (69.2%). It also identifies alternative valid citations and demonstrates generalization ability for cross-domain citation attribution.

- [arXiv](https://www.arxiv.org/abs/2510.17853)
- [Hugging Face paper page](https://huggingface.co/papers/2510.17853)

🎉 Accepted at ACL 2026 Main Conference

## Setup

Run from the repository root.

```bash
pip install -r requirements.txt
```

Required:

- `S2_API_KEY`

Set the model provider key that matches the model you want to use:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `TOGETHER_API_KEY`
- `DEEPSEEK_API_KEY`
- `GOOGLE_API_KEY`

If you use `--local_model`, CiteGuard uses Ollama and does not need a cloud model API key.

## CLI

Single excerpt:

```bash
python -m src.main --model_name gpt-4o --excerpt "Your excerpt with [CITATION]"
```

Dataset:

```bash
python -m src.main --model_name gpt-4o --dataset DATASET.csv --result_path results/run.json
```

Useful options:

- `--result_path`: JSON output file. Existing results are loaded first, and already-processed IDs are skipped.
- `--source_paper_title`: source paper title for the excerpt.
- `--target_paper_title`: gold title for evaluation. Use `[TITLE_SEPARATOR]` for multiple acceptable titles.
- `--skip_citations`: comma-separated titles to exclude.
- `--additional_context`: extra surrounding text provided up front.
- `--no_interactive_context`: disable terminal prompts for more context.
- `--year`: source paper year. Default: `2025`.
- `--temperature`: model temperature. Default: `0.95`.
- `--local_model`: use Ollama instead of a hosted API model.

Example:

```bash
python -m src.main \
  --model_name gpt-4o \
  --source_paper_title "Example Source Paper" \
  --excerpt "Transformer-based retrieval improves grounded generation [CITATION]." \
  --skip_citations "Attention Is All You Need,Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
```

In single-excerpt CLI mode, the agent can ask for more context. Paste one or more lines and press Enter on an empty line to submit, or type `SKIP` to continue without extra context.

## Dataset Format

Expected CSV columns:

- `id`
- `excerpt`
- `year`
- `source_paper_title` (optional)
- `target_paper_title` (optional)

## Web UI

Start the local server:

```bash
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

The web UI supports:

- single-excerpt runs
- dataset runs
- browsing results by ID after a run
- loading an existing `result_path` without re-running the agent via `Load Result Path`

If the `result_path` already exists, the web app loads it first, skips IDs that are already done, and writes the updated results back to that path after the run.

The web UI is not mid-run conversational. If you want to provide extra context, use the `Additional Context` field.

## Output

Each result JSON file contains:

- `metadata`: run configuration
- `results`: one entry per excerpt

Each result entry may include:

- `id`
- `excerpt`
- `selected`
- `status`
- `error`
- `papers`
- `history`
- `duration`
- `is_correct`
- `is_in_search`

## Key Files

- `src/main.py`: CLI entrypoint
- `src/run_main.py`: main execution flow
- `src/retriever/agent.py`: agent loop and tool actions
- `app.py`: Flask backend
- `templates/index.html`: web frontend

## License

- Code: MIT. See `LICENSE`.
- Dataset: CC-BY-4.0. See `LICENSE_DATASET`.

If you use this repository, please cite:

```bibtex
@misc{choi2026citeguardfaithfulcitationattribution,
      title={CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation}, 
      author={Yee Man Choi and Xuehang Guo and Yi R. Fung and Qingyun Wang},
      year={2026},
      eprint={2510.17853},
      archivePrefix={arXiv},
      primaryClass={cs.DL},
      url={https://arxiv.org/abs/2510.17853}, 
}
```

## Acknowledgement

This project builds on [CiteME / CiteAgent](https://github.com/bethgelab/CiteME).
