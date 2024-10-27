import os
import json

def write_metric(metric_name,value,file):
    with open(file,"w") as f:
        json.dump({metric_name:value},f)