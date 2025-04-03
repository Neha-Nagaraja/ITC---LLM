import os
import json
import openai
import anthropic
from dotenv import load_dotenv
from time import sleep
from collections import defaultdict
import re

# ========== Load API keys ==========
load_dotenv()
openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ========== Load questions ==========
with open("aws_questions_with_ids.json") as f:
    questions = json.load(f)

# ========== Prompt template ==========
# ROLE = (
#     "You are a certified AWS Cloud Practitioner Trainer with extensive experience "
#     "in teaching AWS fundamentals. Read the question and respond only with the letter "
#     "of the correct option (a, b, c, or d). "
# )
ROLE = (
    "You are a highly experienced AWS Cloud Practitioner Trainer with 10+ years of industry experience. "
    "Your job is to help students pass the AWS Cloud Practitioner certification by identifying the most accurate, factual, and AWS-aligned answer choices. "
    "Always choose the single best answer based strictly on AWS documentation and best practices. "
    "Do not explain your answer. Only respond with the letter of the correct option (a, b, c, or d)."
)


def build_prompt(q, role=None):
    options_text = "\n".join([f"{k}) {v}" for k, v in q["options"].items()])
    if role:
        return f"{role}\n\nQuestion:\n{q['question']}\n{options_text}\nAnswer:"
    else:
        return f"{q['question']}\n{options_text}\nAnswer:"

def extract_letter(text):
    text = text.strip().lower()

    # 1. Try common patterns like "the correct answer is b)", "option c", "c)", etc.
    patterns = [
        r"\b([a-d])\)",                    # a) b) c)
        r"\boption\s+([a-d])\b",           # option a
        r"answer\s*[:\-]?\s*([a-d])\b",    # answer: b / answer - b
        r"the correct answer is\s+([a-d])\b",  # the correct answer is b
        r"the correct option is\s+([a-d])\b",  # the correct option is b
        r"\b([a-d])\b"                     # just a single a/b/c/d anywhere
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
for idx, q in enumerate(questions):
    question_id = q.get("id", idx + 1)  # fallback if id missing
    for model in ["gpt-3.5-turbo", "claude-3-5-sonnet"]:
        for prompt_type in ["standard", "role"]:
            prompt = build_prompt(q, ROLE if prompt_type == "role" else None)

            if model == "gpt-3.5-turbo":
                response = ask_openai(prompt)
            else:
                response = ask_claude(prompt)

            answer = extract_letter(response or "")
            is_correct = (answer == q["answer"])

            # print(f"Q{question_id} | {model} | {prompt_type} | Ans: {answer or '?'} | {'True ✅' if is_correct else 'False ❌'}")
            print(f"\nQ{question_id} | {model} | {prompt_type}")
            print(f"↪ Raw Response: {response}")
            print(f"↪ Normalized Answer: {answer or '?'} | Correct: {q['answer']} | {'True ✅' if is_correct else 'False ❌'}\n")


            results.append({
                "id": question_id,
                "llm": model,
                "prompt_type": prompt_type,
                "response": response,
                "normalized_answer": answer,
                "correct_answer": q["answer"],
                "is_correct": is_correct
            })

            sleep(1)  # rate limiting buffer

# ========== Save Results to Files ==========
grouped_results = defaultdict(list)
for r in results:
    key = f"{r['llm']}_{r['prompt_type']}"
    grouped_results[key].append(r)

for key, res_list in grouped_results.items():
    filename = f"v1_results_{key}.json"
    with open(filename, "w") as f:
        json.dump(res_list, f, indent=2)
    print(f"Saved {len(res_list)} responses to {filename}")
