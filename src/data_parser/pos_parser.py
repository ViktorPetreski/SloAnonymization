import conllu
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging


def save_to_file(sentences, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write("\n\n".join(sentences))


def split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15):
    """
    Splits documents array into three sets: learning, validation & testing.
    If random seed is given, documents selected for each set are randomly picked (but do not overlap, of course).
    """
    # Note: test_prop is redundant, but it's left in to make it clear this is a split into 3 parts
    test_prop = 1.0 - train_prop - dev_prop

    train_docs, dev_test_docs = train_test_split(documents, test_size=(dev_prop + test_prop))

    dev_docs, test_docs = train_test_split(dev_test_docs, test_size=test_prop/(dev_prop + test_prop))

    save_to_file(train_docs, "../../data/Janes-Tag.conllu/train.conllu")
    save_to_file(dev_docs, "../../data/Janes-Tag.conllu/dev.conllu")
    save_to_file(test_docs, "../../data/Janes-Tag.conllu/test.conllu")

    print(f"{len(documents)} documents split to: training set ({len(train_docs)}), dev set ({len(dev_docs)}) "
                 f"and test set ({len(test_docs)}).")

    return train_docs, dev_docs, test_docs

if __name__ == '__main__':
    dataset_path = Path("../../data/Janes-Tag.conllu/Janes-Tag.conllu/")
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = file.read()

    sentences = data.split("\n\n")
    split_into_sets(sentences)

    # sentences = conllu.parse(data)
    # for sentence in sentences:
    #     print(sentence.metadata)
    #     for token in sentence:
    #         print(token["xpos"])
    #     break
