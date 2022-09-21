from sklearn.model_selection import train_test_split
import os, glob
from pathlib import Path
import shutil
import random
def copy_docs(docs, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    for doc in docs:
        conll_format = os.path.basename(doc).replace("gold_conll", "conll")
        shutil.copy2(doc, os.path.join(dest, conll_format))

def senticoref():
    documents = []
    for file_name in glob.glob("../../data/senticoref_conll/*/*.conll"):
        documents.append(file_name)
    dev_prop = 0.15
    test_prop = 0.15

    train_docs, dev_test_docs = train_test_split(documents, test_size=(dev_prop + test_prop))

    dev_docs, test_docs = train_test_split(dev_test_docs, test_size=test_prop / (dev_prop + test_prop))

    copy_docs(train_docs, "../../data/senticoref_conll-pos-random/train")
    copy_docs(test_docs, "../../data/senticoref_conll-pos-random/test")
    copy_docs(dev_docs, "../../data/senticoref_conll-pos-random/dev")

def core149():
    documents = []
    for file_name in glob.glob("../../data/coref149_conll-enhanced/*.conll"):
        documents.append(file_name)
    dev_prop = 0.15
    test_prop = 0.15

    train_docs, dev_test_docs = train_test_split(documents, test_size=(dev_prop + test_prop))

    dev_docs, test_docs = train_test_split(dev_test_docs, test_size=test_prop / (dev_prop + test_prop))

    copy_docs(train_docs, "../../data/senticoref_conll-random-pos-bkp/train")
    copy_docs(test_docs, "../../data/senticoref_conll-random-pos-bkp/test")
    copy_docs(dev_docs, "../../data/senticoref_conll-random-pos-bkp/dev")

def ontonotes(data_type, num_samples):
    documents = []
    for file_name in glob.glob(f"../../data/conll-formatted-ontonotes-5.0/data/{data_type}/data/english/annotations/*/*/*/*.gold_conll"):
        documents.append(file_name)

    documents = random.sample(documents, k=num_samples)
    copy_docs(documents, f"../../data/senticoref_conll-random-pos-50/{data_type}")


if __name__ == '__main__':
    # ontonotes("train", 350)
    # ontonotes("test", 70)
    # ontonotes("dev", 70)
    documents = []
    for file_name in glob.glob(f"../../data/senticoref_conll-random-pos-50/train/*.conll"):
        documents.append(file_name)
    print(len(documents))