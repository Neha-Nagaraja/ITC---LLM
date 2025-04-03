import os
import json
import openai
import anthropic
from dotenv import load_dotenv
from time import sleep
from collections import defaultdict
import re

# ========== Load API Keys ==========
load_dotenv()
openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ========== Load Questions ==========
with open("aws_questions_with_ids.json") as f:
    eval_questions = json.load(f)

with open("few_shot_examples.json") as f:
    few_shot_examples = json.load(f)

# ========== Prompt Construction ==========
def format_qa(q):
    options = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    return f"Question:\n{q['question']}\n{options}\nAnswer: {q['answer']}"

def build_prompt(q, few_shot_count=0):
    examples = "\n\n".join([format_qa(ex) for ex in few_shot_examples[:few_shot_count]])
    options_text = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    task = f"Question:\n{q['question']}\n{options_text}\nAnswer:"
    return f"{examples}\n\n{task}" if few_shot_count else task

# ========== Answer Extraction ==========
def extract_letter(text):
    text = text.strip().lower()
    patterns = [
        r"\b([a-d])\)", r"\boption\s+([a-d])\b", r"answer\s*[:\-]?\s*([a-d])\b",
        r"the correct answer is\s+([a-d])\b", r"the correct option is\s+([a-d])\b", r"\b([a-d])\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

# ========== LLM Callers ==========
def ask_openai(prompt):
    try:
        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return None

def ask_claude(prompt):
    try:
        res = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()
    except Exception as e:
        print("Claude error:", e)
        return None

# ========== Run Evaluation ==========
results = []

for model in ["gpt-3.5-turbo", "claude-3-5-sonnet"]:
    for few_shot_count in [0, 2, 5, 10]:
        print(f"\nRunning {model} | {few_shot_count}-shot")
        for q in eval_questions:
            prompt = build_prompt(q, few_shot_count)

            response = ask_openai(prompt) if model == "gpt-3.5-turbo" else ask_claude(prompt)
            answer = extract_letter(response or "")
            is_correct = (answer and answer.lower() == q["answer"].lower())

            print(f"Q{q['id']} | {few_shot_count}-shot | {model} | Ans: {answer or '?'} | {'True ✅' if is_correct else 'False ❌'}")

            results.append({
                "id": q["id"],
                "llm": model,
                "shot": few_shot_count,
                "response": response,
                "normalized_answer": answer,
                "correct_answer": q["answer"],
                "is_correct": is_correct
            })

            sleep(1)  # Respect rate limits

# ========== Save Results ==========
grouped = defaultdict(list)
for r in results:
    key = f"{r['llm']}_{r['shot']}shot"
    grouped[key].append(r)

for key, res_list in grouped.items():
    filename = f"results_{key}.json"
    with open(filename, "w") as f:
        json.dump(res_list, f, indent=2)
    print(f"Saved {len(res_list)} results to {filename}")
