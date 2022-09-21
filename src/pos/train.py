from src.data_parser import parser
from sklearn.metrics import classification_report, f1_score
from src import vp_ta_constants
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

tag2id = {}
id2tag = {}


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(id2tag.keys())
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    mlb = MultiLabelBinarizer()
    true_predictions = mlb.fit_transform(y=true_predictions)
    true_labels = mlb.fit_transform(y=true_labels)
    return classification_report(true_labels, true_predictions, target_names=list(tag2id.keys()), output_dict=True)


if __name__ == '__main__':
    train, test, dev = parser.get_ssj500k("xpos")
    all_tags = set()
    for x in dev["tags"]:
        all_tags = all_tags.union(x)
    # tag2id = {tag: id for id, tag in enumerate(all_tags)}
    tags = dev["tags"]
    # for chunk in dev["tags"]:
    #     tags.append(list(map(lambda x: tag2id[x], chunk)))
    mlb = MultiLabelBinarizer()
    tags = mlb.fit_transform(y=tags)
    print(classification_report(tags, tags))
    # print(len(train["sentences"]), len(test["sentences"]), len(dev["sentences"]))
    # print(dev["tags"][0], test["tags"][0])