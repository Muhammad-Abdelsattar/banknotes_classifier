import json

with open("diff.json","r") as f:
    metrics = json.load(f)

value = "less"
try:
    diff = metrics["reports/evaluation.json"]["accuracy"]["diff"]
    if(diff >= -0.05):
        value = "acceptable"
except KeyError:
    value = "NO Value"

with open("diff.txt","w") as f:
    f.write(value)