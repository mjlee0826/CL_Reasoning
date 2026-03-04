from datasets import load_dataset
import json

# Load mGSM English split
dataset = load_dataset("sbintuitions/MGSM_en")

# Merge all splits into one list
all_data = []
for split in dataset.keys():
    all_data.extend(dataset[split])

# Save everything into one JSON file
with open("./data/mgsm_en.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

# Show an example entry
print(all_data[0])
