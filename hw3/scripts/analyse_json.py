import json
from collections import defaultdict

def main():
    with open("re_evaluated_results.json", "r") as infile:
        data = json.load(infile)

    stats = defaultdict(lambda: {"Jailbreak Success": 0, "Defense Success": 0})

    for entry in data:
        key = f"{entry['prompt_name']} | {entry['model']}"
        clean_result = entry['re_evaluated_result'].split(" --> ")[0] 
        stats[key][clean_result] += 1

    print("\n--- Jailbreak Summary ---\n")

    for key, result_counts in stats.items():
        jailbreaks = result_counts["Jailbreak Success"]
        defenses = result_counts["Defense Success"]
        total = jailbreaks + defenses
        print(f"{key} --> Jailbreak: {jailbreaks} / {total}")

if __name__ == "__main__":
    main()
