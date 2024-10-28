import json

with open("diff.json","r") as f:
    metrics = json.load(f)

diff = metrics["evaluation.json"]["accuracy"]["diff"]
value = "less"
if(diff >= 0):
    value = "greater"
with open("diff.txt","w") as f:
    f.write(value)