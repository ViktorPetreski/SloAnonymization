import pandas as pd
import glob, os

if __name__ == '__main__':
    res = {
        "bsnlp": [],
        "ssj500k": [],
        "combined" : []
    }

    for file in glob.glob("../../results/ner/*_new*.csv"):
        df = pd.read_csv(file)
        micro_score = df.iloc[-2, -1]
        macro_score = df.iloc[-1, -1]
        if "bsnlp" in file:
            res["bsnlp"].append({
                "name": file,
                "micro_score": micro_score,
                "macro_score": macro_score,
                "ratio": macro_score/micro_score
            })
        elif "ssj500k" in file:
            res["ssj500k"].append({
                "name": file,
                "micro_score": micro_score,
                "macro_score": macro_score,
                "ratio": macro_score/micro_score
            })
        elif "combined" in file:
            res["combined"].append({
                "name": file,
                "micro_score": micro_score,
                "macro_score": macro_score,
                "ratio": macro_score/micro_score
            })
    for key,value in res.items():
        newlist = sorted(value, key=lambda k: k['micro_score'], reverse=True)
        print(newlist)