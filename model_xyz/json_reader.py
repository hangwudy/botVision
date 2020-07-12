import json
import csv
import pandas as pd


path = "bigdata/output/history_vgg16.json"

with open(path, "r") as f:
    history = json.load(f)

print(history.keys())
df = pd.DataFrame(history)
print(df)

df.to_csv("bigdata/output/history.csv", encoding="utf-8")
