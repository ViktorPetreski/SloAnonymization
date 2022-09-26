import json

# from src.allennlp.readability_metrics import *
# import nltk
import glob
from collections import defaultdict

from bs4 import BeautifulSoup
import os
import requests
import pandas as pd
import pickle

# from src.py_models.Pipeline import  Pipeline

def get_scores():
    results = pickle.load(open("readability.p", "rb"))
    averages = pickle.load(open("averages.p", "rb"))
    # pipeline = Pipeline("", "readable")
    for pat in glob.glob("../../data/coref149/*.tcf"):
        file_name = os.path.basename(pat)
        print(file_name)
        if file_name in results.keys() or file_name == "ssj79.517.tcf" or file_name == "ssj73.459.tcf":
            continue
        try:
            with open(pat, "r") as file:
                bs = BeautifulSoup(file.read(), "lxml")
                texts = bs.findAll("tc:text")
                text = "".join([t.text for t in texts]).replace(" , ", ", ").replace(" . ", ". ").replace(" ; ", "; ").replace(" ? ", "? ").replace(" ! ", "! ").replace(" : ", ": ")

                data = {
                    "text": text,
                    "mode": ""
                }
                pseudo_text = requests.post("http://localhost:5050/anonymize", json=data).json()["text"]

                data = {
                    "text": text,
                    "mode": "low"
                }
                low_text = requests.post("http://localhost:5050/anonymize", json=data).json()["text"]

                data = {
                    "text": text,
                    "mode": "medium"
                }
                medium_text = requests.post("http://localhost:5050/anonymize", json=data).json()["text"]

                data = {
                    "text": text,
                    "mode": "high"
                }
                high_text = requests.post("http://localhost:5050/anonymize", json=data).json()["text"]

                results[file_name] = {
                    "orig_flesch": flesch_reading_ease(text),
                    "orig_gunning_fog": gunning_fog(text),
                    "orig_SMOG": smog_index(text),
                    "orig_Dale-Chall": dale_chall_readability_score(text),
                    "pseudo_flesch": flesch_reading_ease(pseudo_text),
                    "pseudo_gunning_fog": gunning_fog(pseudo_text),
                    "pseudo_SMOG": smog_index(pseudo_text),
                    "pseudo_Dale-Chall": dale_chall_readability_score(pseudo_text),
                    "low_flesch": flesch_reading_ease(low_text),
                    "low_gunning_fog": gunning_fog(low_text),
                    "low_SMOG": smog_index(low_text),
                    "low_Dale-Chall": dale_chall_readability_score(low_text),
                    "medium_flesch": flesch_reading_ease(medium_text),
                    "medium_gunning_fog": gunning_fog(medium_text),
                    "medium_SMOG": smog_index(medium_text),
                    "medium_Dale-Chall": dale_chall_readability_score(medium_text),
                    "high_flesch": flesch_reading_ease(high_text),
                    "high_gunning_fog": gunning_fog(high_text),
                    "high_SMOG": smog_index(high_text),
                    "high_Dale-Chall": dale_chall_readability_score(high_text),
                }

                averages = {
                    "orig_flesch": results[file_name]["orig_flesch"] + averages["orig_flesch"],
                    "orig_gunning_fog": results[file_name]["orig_gunning_fog"] + averages["orig_gunning_fog"],
                    "orig_SMOG": results[file_name]["orig_SMOG"] + averages["orig_SMOG"],
                    "orig_Dale-Chall": results[file_name]["orig_Dale-Chall"] + averages["orig_Dale-Chall"],
                    "pseudo_flesch": results[file_name]["pseudo_flesch"] + averages["pseudo_flesch"],
                    "pseudo_gunning_fog": results[file_name]["pseudo_gunning_fog"] + averages["pseudo_gunning_fog"],
                    "pseudo_SMOG": results[file_name]["pseudo_SMOG"] + averages["pseudo_SMOG"],
                    "pseudo_Dale-Chall": results[file_name]["pseudo_Dale-Chall"] + averages["pseudo_Dale-Chall"],
                    "low_flesch": results[file_name]["low_flesch"] + averages["low_flesch"],
                    "low_gunning_fog": results[file_name]["low_gunning_fog"] + averages["low_gunning_fog"],
                    "low_SMOG": results[file_name]["low_SMOG"] + averages["low_SMOG"],
                    "low_Dale-Chall": results[file_name]["low_Dale-Chall"] + averages["low_Dale-Chall"],
                    "medium_flesch": results[file_name]["medium_flesch"] + averages["medium_flesch"],
                    "medium_gunning_fog": results[file_name]["medium_gunning_fog"] + averages["medium_gunning_fog"],
                    "medium_SMOG": results[file_name]["medium_SMOG"] + averages["medium_SMOG"],
                    "medium_Dale-Chall": results[file_name]["medium_Dale-Chall"] + averages["medium_Dale-Chall"],
                    "high_flesch": results[file_name]["high_flesch"] + averages["high_flesch"],
                    "high_gunning_fog": results[file_name]["high_gunning_fog"] + averages["high_gunning_fog"],
                    "high_SMOG": results[file_name]["high_SMOG"] + averages["high_SMOG"],
                    "high_Dale-Chall": results[file_name]["high_Dale-Chall"] + averages["high_Dale-Chall"],
                }

                pickle.dump(results, open("readability.p", "wb"))
                pickle.dump(averages, open("averages.p", "wb"))
        except:
            continue

    print(json.dumps(results, indent=4))
    pd.DataFrame.from_dict(results)
    print(pd)

if __name__ == '__main__':
    averages = pickle.load(open("averages.p", "rb"))
    results = pickle.load(open("readability.p", "rb"))
    print(len(results.keys()))
    df = pd.DataFrame.from_dict(averages, orient="index")
    for col in df.columns:
        print(col)
    print(df)
    df["Avg"] = df[0] / len(results.keys())
    df = df.drop(columns=[0])

    print(df.to_latex())