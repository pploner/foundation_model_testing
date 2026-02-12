import json

# Load the JSON file
with open("src/utils/nEvents_scan/file_event_counts.json") as f:
    data = json.load(f)

# Sum events per class
totals = {}
for class_name, files in data.items():
    totals[class_name] = sum(files.values())

with open("src/utils/nEvents_scan/class_totals.json", "w") as f:
    json.dump(totals, f, indent=2)
