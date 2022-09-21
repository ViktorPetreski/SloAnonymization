import glob
import os.path

from conllu import parse
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict

project_location = "../../data/BSNLP/datasets"


def check_existing_data(dataset_name, task):
    all_train_path = Path(f"../../data/datasets/{dataset_name}/{task}/train.pickle")
    all_test_path = Path(f"../../data/datasets/{dataset_name}/{task}/test.pickle")
    all_dev_path = Path(f"../../data/datasets/{dataset_name}/{task}/dev.pickle")
    c1 = all_train_path.is_file()
    c2 = all_test_path.is_file()
    c3 = all_dev_path.is_file()
    if c1 and c2 and c3:
        all_train = pickle.load(open(all_train_path, "rb"))
        all_test = pickle.load(open(all_test_path, "rb"))
        all_dev = pickle.load(open(all_dev_path, "rb"))
        return all_train, all_test, all_dev
    return False


def conllu_to_conll(file_name, task="ner"):
    res = {"sentences": [], "tags": []}
    with open(f'{file_name}.conllu', 'rb') as file:
        content = file.read().decode("utf-8")
        sentences = parse(content)

    for sentence in sentences:
        text = sentence.metadata["text"]
        tags = []
        words = []
        for token in sentence:
            tag = ""
            if task == "ner":
                tag = list(token['misc'].keys())[0]
                tag = tag.upper()
            elif task == "xpos":
                tag = token["upos"]
                if token["upos"] == "PRON":
                    tag = token["xpos"][:4]
                elif token["xpos"] is not None and len(token["xpos"]) < 2:
                    tag = token["xpos"]
                elif token["upos"] in ["NOUN", "PROPN"]:
                    tag = token["xpos"][:3]
                else:
                    tag = "O"
            elif task == "upos":
                tag = token["upos"]
            tags.append(tag)
            words.append(token['form'])
        res["tags"].append(tags)
        res["sentences"].append(words)
    #                 file.write(bytes(f"{token['form']}\t{token['upos']}\t{token['xpos']}\t{ner}\n", "utf-8"))
    #             file.write(b"\n")
    return res


def get_jos1m(task):
    all_train_path = Path(f"../../data/datasets/jos1m/{task}/train.pickle")
    all_test_path = Path(f"../../data/datasets/jos1m/{task}/test.pickle")
    all_dev_path = Path(f"../../data/datasets/jos1m/{task}/dev.pickle")
    result = check_existing_data("jos1m", task)
    if result is not False:
        return result

    dev = conllu_to_conll("../../data/jos1M.conllu/dev", task)
    train = conllu_to_conll("../../data/jos1M.conllu/train", task)
    test = conllu_to_conll("../../data/jos1M.conllu/test", task)

    pickle.dump(train, open(all_train_path, "wb"))
    pickle.dump(test, open(all_test_path, "wb"))
    pickle.dump(dev, open(all_dev_path, "wb"))

    return train, test, dev


def get_janes_tag(task):
    all_train_path = Path(f"../../data/datasets/janes_tag/{task}/train.pickle")
    all_test_path = Path(f"../../data/datasets/janes_tag/{task}/test.pickle")
    all_dev_path = Path(f"../../data/datasets/janes_tag/{task}/dev.pickle")
    result = check_existing_data("janes_tag", task)
    if result is not False:
        return result

    dev = conllu_to_conll("../../data/Janes-Tag.conllu/dev", task)
    train = conllu_to_conll("../../data/Janes-Tag.conllu/train", task)
    test = conllu_to_conll("../../data/Janes-Tag.conllu/test", task)

    pickle.dump(train, open(all_train_path, "wb"))
    pickle.dump(test, open(all_test_path, "wb"))
    pickle.dump(dev, open(all_dev_path, "wb"))

    return train, test, dev


def get_ssj500k(task):
    all_train_path = Path(f"../../data/datasets/ssj500k/{task}/train.pickle")
    all_test_path = Path(f"../../data/datasets/ssj500k/{task}/test.pickle")
    all_dev_path = Path(f"../../data/datasets/ssj500k/{task}/dev.pickle")
    result = check_existing_data("ssj500k", task)
    if result is not False:
        return result
    train = []
    test = []
    dev = []
    if task == "ner":
        dev = conllu_to_conll(f"{project_location}/ssj500k/dev_ner", task)
        train = conllu_to_conll(f"{project_location}/ssj500k/train_ner", task)
        test = conllu_to_conll(f"{project_location}/ssj500k/test_ner", task)
    elif task in ["xpos", "upos"]:
        dev = conllu_to_conll(f"{project_location}/ssj500k/ssj500k.conllu/sl_ssj-ud_v2.4-dev", task)
        train = conllu_to_conll(f"{project_location}/ssj500k/ssj500k.conllu/sl_ssj-ud_v2.4-train", task)
        test = conllu_to_conll(f"{project_location}/ssj500k/ssj500k.conllu/sl_ssj-ud_v2.4-test", task)

    pickle.dump(train, open(all_train_path, "wb"))
    pickle.dump(test, open(all_test_path, "wb"))
    pickle.dump(dev, open(all_dev_path, "wb"))

    return train, test, dev


def join_sentences_from_df(input_data: pd.DataFrame, task):
    tokens = []
    tags = []  # NER/POS tags
    kontra = "ner"
    if task in ["xpos", "upos"]:
        task = "xpos" if task == "upos" else "upos"
        kontra = "upos" if task == "xpos" else "xpos"
    for (_, sentence), data in input_data.groupby(["docId", "sentenceId"]):
        sentence_tokens = []
        sentence_tags = []
        for id, word_row in data.iterrows():
            word_tokens = str(word_row["text"])
            sentence_tokens.append(word_tokens)
            if task in ["xpos", "upos"]:
                tag = word_row[kontra]
                if word_row[kontra] == "PRON":
                    tag = word_row[task][:4]
                elif len(word_row[task]) < 2:
                    tag = word_row[task]
                elif word_row[kontra] in ["NOUN", "PROPN"]:
                    tag = word_row[task][:3]
                sentence_tags.append(tag)
            elif task == "ner":
                sentence_tags.append(word_row[task])

        tokens.append(sentence_tokens)
        tags.append(sentence_tags)

    return tokens, tags


def get_bsnlp(task='ner'):
    datasets = [
        f"{project_location}/bsnlp/asia_bibi",
        f"{project_location}/bsnlp/brexit",
        f"{project_location}/bsnlp/ec",
        f"{project_location}/bsnlp/nord_stream",
        f"{project_location}/bsnlp/other",
        f"{project_location}/bsnlp/trump",
        f"{project_location}/bsnlp/ryanair",
        # f"{project_location}/bsnlp/us_election_2020",
    ]

    all_train_path = Path(f"../../data/datasets/bsnlp/{task}/train.pickle")
    all_test_path = Path(f"../../data/datasets/bsnlp/{task}/test.pickle")
    all_dev_path = Path(f"../../data/datasets/bsnlp/{task}/dev.pickle")

    result = check_existing_data("bsnlp", task)
    if result is not False:
        return result

    all_dev = {
        "sentences": [],
        "tags": []
    }
    all_train = {
        "sentences": [],
        "tags": []
    }

    all_test = {
        "sentences": [],
        "tags": []
    }

    for dataset in datasets:
        dev = pd.read_csv(f"{dataset}/splits/sl/dev_sl.csv")
        train = pd.read_csv(f"{dataset}/splits/sl/train_sl.csv")
        test = pd.read_csv(f"{dataset}/splits/sl/test_sl.csv")

        dev_tokens, dev_tags = join_sentences_from_df(dev, task)
        test_tokens, test_tags = join_sentences_from_df(test, task)
        train_tokens, train_tags = join_sentences_from_df(train, task)

        all_dev["sentences"].extend(dev_tokens)
        all_dev["tags"].extend(dev_tags)

        all_train["sentences"].extend(train_tokens)
        all_train["tags"].extend(train_tags)

        all_test["sentences"].extend(test_tokens)
        all_test["tags"].extend(test_tags)

    pickle.dump(all_train, open(all_train_path, "wb"))
    pickle.dump(all_test, open(all_test_path, "wb"))
    pickle.dump(all_dev, open(all_dev_path, "wb"))

    return all_train, all_test, all_dev


def get_combined(task="ner"):
    bsnlp_train, bsnlp_test, bsnlp_dev = get_bsnlp(task)
    ssj_train, ssj_test, ssj_dev = get_ssj500k(task)

    if task in ["upos", "xpos"]:
        jos1m_train, jos1m_test, jos1m_dev = get_jos1m(task)
        janes_tag_train, janes_tag_test, janes_tag_dev = get_janes_tag(task)

    all_train_path = Path(f"../../data/datasets/combined/{task}/train.pickle")
    all_test_path = Path(f"../../data/datasets/combined/{task}/test.pickle")
    all_dev_path = Path(f"../../data/datasets/combined/{task}/dev.pickle")

    result = check_existing_data("combined", task)
    if result is not False:
        return result

    all_train = {
        "sentences": [*bsnlp_train["sentences"], *ssj_train["sentences"]],
        "tags": [*bsnlp_train["tags"], *ssj_train["tags"]]
    }

    if task in ["xpos", "upos"]:
        all_train["sentences"].extend([*jos1m_train["sentences"], *janes_tag_train["sentences"]])
        all_train["tags"].extend([*jos1m_train["tags"], *janes_tag_train["tags"]])

    pickle.dump(all_train, open(all_train_path, "wb"))

    all_test = {
        "sentences": [*bsnlp_test["sentences"], *ssj_test["sentences"]],
        "tags": [*bsnlp_test["tags"], *ssj_test["tags"]]
    }

    if task in ["xpos", "upos"]:
        all_test["sentences"].extend([*jos1m_test["sentences"], *janes_tag_test["sentences"]])
        all_test["tags"].extend([*jos1m_test["tags"], *janes_tag_test["tags"]])

    pickle.dump(all_test, open(all_test_path, "wb"))

    all_dev = {
        "sentences": [*bsnlp_dev["sentences"], *ssj_dev["sentences"]],
        "tags": [*bsnlp_dev["tags"], *ssj_dev["tags"]]
    }

    if task in ["xpos", "upos"]:
        all_dev["sentences"].extend([*jos1m_dev["sentences"], *janes_tag_dev["sentences"]])
        all_dev["tags"].extend([*jos1m_dev["tags"], *janes_tag_dev["tags"]])

    pickle.dump(all_dev, open(all_dev_path, "wb"))

    return all_train, all_test, all_dev


def parse_results(res):
    result = defaultdict(list)
    for key, value in res.items():
        key_parts = key.split("_")
        if len(key_parts) == 3 and key_parts[2] != "number" and key_parts[2] != "accuracy":
            result[key_parts[1]].append(round(value, 3))

    columns = ["Precision", "Recall", "F1-Score"]

    df = pd.DataFrame.from_dict(result, orient="index", columns=columns)
    print(df)
    return df


def combine_coref_si(file_name):
    res = {"sentences": [], "tags": []}
    with open(f'{file_name}.conllu', 'rb') as file:
        content = file.read().decode("utf-8")
        sentences = parse(content)

    for file in glob.glob("../../data/coref149_conll/*.conll"):
        with(open(file, "r")) as conll:
            basename = os.path.basename(file).replace(".conll", "")
            for sentence in sentences:
                text = sentence.metadata["text"]
                new_sentences = []
                ner_tags = []
                words = []
                # print(basename, sentence.metadata["sent_id"])
                if basename in sentence.metadata["sent_id"]:
                    print(basename, sentence.metadata["sent_id"])
                    lines = []
                    for line in conll.readlines():
                        if len(line) > 0 and (line.startswith("#begin") or line.startswith("#end")):
                            if line.startswith("#begin"):
                                new_sentences.append(line)
                            continue
                        lines.append(line)
                    for line, token in zip(lines, sentence):
                        parts = line.split()
                        is_coref = False
                        if parts[3] == token["form"]:
                            parts[4] = token["xpos"]
                            parts[6] = token["lemma"]
                            coref = parts[-1]
                            ner_tag = token['misc']["NER"].upper()
                            ner_tag = ner_tag if ner_tag == "O" else ner_tag.split("-")[1]
                            if ner_tag != "O":
                                if "(" in coref and ")" not in coref:
                                    parts[-2] = f"({ner_tag}*"  if ner_tag != "O" else "*"
                                    is_coref = True
                                elif "(" in coref and ")" in coref:
                                    parts[-2] = f"({ner_tag})" if ner_tag != "O" else "*"
                                elif "(" not in coref and ")" in coref:
                                    parts[-2] = f"{ner_tag})"  if ner_tag != "O" else "*"
                                    is_coref = False
                            new_sentences.append("\t\t\t".join(parts))
                            # new_sentences.append("\n")
                    new_sentences.append("\n#end document")
                    enhanced_coref_path = Path("../../data/coref149_conll-enhanced").mkdir(parents=True, exist_ok=True)
                    with(open(f"../../data/coref149_conll-enhanced/{basename}.conll", "w")) as dest:
                        dest.writelines("\n".join(new_sentences))
                    break


if __name__ == '__main__':
    combine_coref_si(f"{project_location}/ssj500k/ssj500k-morpho")
