import os
import json
import openai
import anthropic
from collections import Counter, defaultdict
from dotenv import load_dotenv
from time import sleep

# ========== Load Keys ==========
load_dotenv()
openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ========== Load Questions ==========
with open("aws_questions_with_ids.json") as f:
    questions = json.load(f)

# ========== Prompt Formatter ==========
def build_prompt(q):
    options = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    return f"{q['question']}\n{options}\nAnswer:"

# ========== Normalize Answer ==========
def extract_letter(text):
    import re
    text = text.strip().lower()
    patterns = [
        r"\b([a-d])\)", r"\boption\s+([a-d])\b",
        r"answer\s*[:\-]?\s*([a-d])\b", r"\b([a-d])\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

# ========== LLM Functions ==========
def ask_gpt(prompt):
    try:
        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
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
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()
    except Exception as e:
        print("Claude error:", e)
        return None

# ========== Self-Consistency Evaluation ==========
def run_self_consistency(model_name, ask_func, n_runs=3):
    results = []
    print(f"\n🔁 Running Self-Consistency for {model_name} with {n_runs} runs per question")

    for q in questions:
        prompt = build_prompt(q)
        responses = []

        for _ in range(n_runs):
            response = ask_func(prompt)
            norm = extract_letter(response or "")
            responses.append({
                "raw_response": response,
                "normalized": norm
            })
            sleep(1)

        # Majority voting
        norm_answers = [r["normalized"] for r in responses if r["normalized"]]
        majority = Counter(norm_answers).most_common(1)
        voted_answer = majority[0][0] if majority else None
        is_correct = voted_answer == q["answer"]

        print(f"Q{q['id']} | Final: {voted_answer or '?'} | Correct: {q['answer']} | {'✅' if is_correct else '❌'}")

        results.append({
            "id": q["id"],
            "llm": model_name,
            "true_answer": q["answer"],
            "voted_answer": voted_answer,
            "is_correct": is_correct,
            "runs": responses
        })

    # Save results
    filename = f"results_{model_name}_selfconsistency.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {filename}")

# ========== Run ==========
run_self_consistency("gpt-3.5-turbo", ask_gpt, n_runs=3)
run_self_consistency("claude-3-5-sonnet", ask_claude, n_runs=3)
