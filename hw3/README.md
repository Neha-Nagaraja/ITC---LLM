# Homework 3 – Part 1: Jailbreak Prompt Testing Framework

This repository contains scripts and prompts used to evaluate the effectiveness of original jailbreak prompts against two large language models (LLMs): **GPT-3.5-Turbo** and **Claude 3-7 Sonnet**.

## 📁 Folder Structure

### `/scripts/`

- **`test.py`**  
  Runs five jailbreak attempts per prompt across both LLMs. Captures and evaluates model responses using GPT-based judgment and saves the output in `results.json`.

- **`re-val.py`**  
  Re-evaluates the model responses from `results.json` using a stricter LLM-based prompt. Labels each response as either `"Jailbreak Success"` or `"Defense Success"` and stores results in `re_evaluated_results.json`.

- **`summary.py`**  
  Aggregates and prints a summary of jailbreak success rates for each prompt-model pair based on the re-evaluated results.

---

### `/prompts/`

Contains 5 original jailbreak prompts developed for this assignment:

- `scenario_nesting.txt`
- `context_based.txt`
- `code_injection.txt`
- `prompt_rewriting.txt`
- `low_resource_lang.txt`

Each prompt was designed using indirect strategies (e.g., simulation, ciphering, or roleplay) to test model robustness without explicit harmful intent.

---

### `/results/`

- **`results.json`** – Raw outputs from the first evaluation run.
- **`re_evaluated_results.json`** – Cleaned and stricter second-pass evaluations used for final analysis.

---

## 🧪 Goal

The scripts help assess:
- Which prompts are more effective in bypassing LLM safeguards
- How consistently each model resists or fails under repeated jailbreak attempts

Each test was run **5 times per prompt per model**, and results were judged by an LLM using a classification scheme: `"Jailbreak Success"` or `"Defense Success"`.


