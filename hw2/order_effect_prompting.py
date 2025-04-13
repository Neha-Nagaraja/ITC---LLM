import os
import json
import openai
from dotenv import load_dotenv
from time import sleep

# ========== Load API Keys ==========
load_dotenv()
openai_client = openai.OpenAI()

# ========== Load Questions ==========
with open("aws_questions_with_ids.json") as f:
    questions = json.load(f)

with open("few_shot_examples.json") as f:
    few_shot_examples = json.load(f)

# ========== Define 3 Orders ==========
orderA = few_shot_examples[:5]  # Original order [1,2,3,4,5]
orderB = list(reversed(orderA))  # Reversed order [5,4,3,2,1]
orderC = [orderA[i] for i in [2, 0, 3, 1, 4]]  # Shuffled order [3,1,4,2,5]

orders = {
    "orderA": orderA,
    "orderB": orderB,
    "orderC": orderC
}

# ========== Prompt Construction ==========
def format_qa(q):
    options = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    return f"Question:\n{q['question']}\n{options}\nAnswer: {q['answer']}"

def build_prompt(q, examples):
    example_text = "\n\n".join([format_qa(ex) for ex in examples])
    options = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    return f"{example_text}\n\nQuestion:\n{q['question']}\n{options}\nAnswer:"

def extract_letter(text):
    import re
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

# ========== Run Experiments ==========
for order_name, example_list in orders.items():
    print(f"\nRunning GPT-3.5-Turbo with {order_name}")
    results = []

    for q in questions:
        prompt = build_prompt(q, example_list)
        try:
            res = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            response = res.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI error:", e)
            response = None

        normalized = extract_letter(response or "")
        is_correct = (normalized and normalized.lower() == q["answer"].lower())

        print(f"Q{q['id']} | {order_name} | Ans: {normalized or '?'} | {'True ✅' if is_correct else 'False ❌'}")

        results.append({
            "id": q["id"],
            "llm": "gpt-3.5-turbo",
            "order": order_name,
            "response": response,
            "normalized_answer": normalized,
            "correct_answer": q["answer"],
            "is_correct": is_correct
        })

        sleep(1)  # Rate limit buffer

    # Save results
    filename = f"results_gpt-3.5-turbo_{order_name}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {filename}")
