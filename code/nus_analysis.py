import json
from collections import Counter
from operator import itemgetter
with open("nus_wide/train.json") as f:
    dat_file = json.load(f)

print(dat_file.keys())

sample_dict = {}

print(dat_file['labels'])
print(dat_file['samples'][0])

for sample in dat_file['samples']:
    for label in sample['image_labels']:
        if label in sample_dict:
            sample_dict[label] += 1
        else:
            sample_dict[label] = 1


sample_dict = [{"name": k, "count": v} for k, v in sample_dict.items()]

sample_dict = sorted(sample_dict, key=itemgetter('count'), reverse=True)[0:20]

for i in sample_dict:
    print(i['name'])

print(sample_dict[0:20])