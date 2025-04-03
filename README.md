# ITC---LLM

# ITC-599-004 — Homework Assignment #2: Prompting Techniques with LLMs  
**Topic:** Foundations and Applications of Large Language Models  
**Author:** Neha Nagaraja  
**Submission Date:** April 3, 2025  

---

##  Repository Structure

This repository contains the implementation, evaluation, and results for all four questions in Homework Assignment #2, which involved experimenting with various prompting techniques using two LLMs: **GPT-3.5-Turbo** and **Claude 3.5 Sonnet (20241022)**.

---

##  LLMs Used
- `gpt-3.5-turbo` — via OpenAI API
- `claude-3-5-sonnet-20241022` — via Anthropic API

All API keys are loaded via `.env`.

---

##  Dataset

- `aws_questions_with_ids.json`:  
  Contains 31 AWS Cloud Practitioner MCQs, including options, correct answers, and explanations.

---

##  Question-wise File Breakdown

###  Question 1: Role Assignment Prompting
**Goal:** Compare zero-shot vs. role-assigned prompting  
**Main Script:**  
- `role_prompting.py`

**Output Files:**  
- `results_gpt-3.5-turbo_standard.json`  
- `results_gpt-3.5-turbo_role.json`  
- `results_claude-3-5-sonnet_standard.json`  
- `results_claude-3-5-sonnet_role.json`

---

###  Question 2: Zero-Shot and Few-Shot Prompting  
**Goal:** Compare performance across 0, 2, 5, and 10-shot settings  
**Main Scripts:**  
- `few_shot_prompting.py`  
- `few_shot_examples.json` (contains few-shot examples used)

**Output Files:**  
- `results_gpt-3.5-turbo_0shot.json`  
- `results_gpt-3.5-turbo_2shot.json`  
- `results_gpt-3.5-turbo_5shot.json`  
- `results_gpt-3.5-turbo_10shot.json`  
- `results_claude-3-5-sonnet_0shot.json`  
- `results_claude-3-5-sonnet_2shot.json`  
- `results_claude-3-5-sonnet_5shot.json`  
- `results_claude-3-5-sonnet_10shot.json`

---

###  Question 3: Sample Order in Few-Shot Prompting  
**Goal:** Test whether changing the order of few-shot examples affects performance  
**Main Script:**  
- `order_effect_prompting.py`

**Output Files:**  
- `results_gpt-3.5-turbo_orderA.json`  
- `results_gpt-3.5-turbo_orderB.json`  
- `results_gpt-3.5-turbo_orderC.json`

---

###  Question 4: Self-Consistency Prompting  
**Goal:** Ask each question 3 times and apply majority voting  
**Main Script:**  
- `self_consistency_prompting.py`

**Output Files:**  
- `results_gpt-3.5-turbo_selfconsistency.json`  
- `results_claude-3-5-sonnet_selfconsistency.json`

---

##  Notes
- All response files include normalized answers, correctness flags, and LLM responses.
- Evaluation summaries were printed to console and used for analysis in the final report.
- Temperature used: `0.0` (for standard/few-shot/role) and `0.7` (for self-consistency).

---

##  Acknowledgments
This repository is part of coursework for ITC-599-004 at Northern Arizona University. All code and data are shared for academic purposes only.

