# read files in results/0/no_model_name_available/no_revision_available/
# and aggregate the results

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# read all files in the directory

# read all files


def get_data(str):
    json_data = []
    for i in range(13):
        try:
            with open(f"/Users/shrenikborad/pless/csci6515_nlu/finalproject/new_data/results_latest_2/{i}/no_model_name_available/no_revision_available/{str}.json", "r") as f:
                data = json.load(f)
                json_data.append(data)
        except FileNotFoundError:
            print(f"File not found for layer {i} for task {str}")
            continue
    final_scores = []
    for data in json_data:
        main_scores = []
        for i in data["scores"]["test"]:
            main_scores.append(i["main_score"])
        final_scores.append(sum(main_scores) / len(main_scores))
    return final_scores


# aggregate the results
all_tasks = {
    "Classification": ["AmazonCounterfactualClassification", "AmazonReviewsClassification", "Banking77Classification", "EmotionClassification", "MTOPDomainClassification", "MTOPIntentClassification", "MassiveIntentClassification", "MassiveScenarioClassification", "ToxicConversationsClassification", "TweetSentimentExtractionClassification"],

    "Clustering": ["ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S", "RedditClustering",
                   "StackExchangeClustering", "TwentyNewsgroupsClustering"],
    "Reranking": ["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions"],
    "STS": ["BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STSBenchmark"],
    "Pair_Classification": ["SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus"],
}

# make a bar plot of the scores
#

tasks = [tasks for tasks in all_tasks.values()]
paths = [item for sublist in tasks for item in sublist]

scoresByTask = {}
for path in paths:
    final_scores = get_data(path)
    scoresByTask[path] = final_scores

    print(scoresByTask[path])
mean_scores_by_category = {}
for category, tasks in all_tasks.items():
    scoresByTask[category] = []
    for layer in range(13):
        layer_scores = []
        try:
            for task in tasks:
                layer_scores.append(scoresByTask[task][layer])
            scoresByTask[category].append(
                sum(layer_scores) / len(layer_scores))
        except:
            print(f"Error in layer {layer} for category {category}")
            continue


# make line plot of the scores

# make csv
scoresByTask = {k: v for k, v in scoresByTask.items() if len(v) > 0}
df = pd.DataFrame(scoresByTask)
df.to_csv("results_latest_new_2.csv", index=False)

for task in scoresByTask.keys():
    plt.plot(scoresByTask[task], label=task)
plt.show()
