import json

with open('nus_wide/test.json') as f:
    train = json.load(f)
    print(train.keys())