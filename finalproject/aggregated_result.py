# read files in results/0/no_model_name_available/no_revision_available/
# and aggregate the results

import os
import json

# read all files in the directory

# read all files

def get_data(str) :
    json_data = []
    for i in range(12):
        with open(f"/Users/shrenikborad/pless/csci6515_nlu/results/{i}/no_model_name_available/no_revision_available/{str}.json", "r") as f:
            data = json.load(f)
            json_data.append(data)
    final_scores = []
    for data in json_data:
        score = data["scores"]["test"][0]["main_score"]
        final_scores.append(score)
    return final_scores


# aggregate the results


# make a bar plot of the scores
paths = ["BIOSSES","SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STSBenchmark", "Banking77Classification"]

scoresByTask= {}
for path in paths:
    final_scores = get_data(path)
    scoresByTask[path] = final_scores
    print(scoresByTask[path])
# make line plot of the scores
import matplotlib.pyplot as plt

for task in scoresByTask.keys():
    plt.plot(scoresByTask[task], label=task)
plt.legend()
plt.show()
