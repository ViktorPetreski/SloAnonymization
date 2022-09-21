import numpy as np
import torch
import transformers


class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def generate_dataset(encodings, labels) -> WNUTDataset:
    encodings.pop("offset_mapping")  # we don't want to pass this to the model
    dataset = WNUTDataset(encodings, labels)
    return dataset


def generate_encodings(tokenizer: transformers.AutoTokenizer.from_pretrained, texts):
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=128
    )
    return encodings


def encode_tags(tags, encodings, unique_tags) -> list[list]:
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    enc_count = 0
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)[:128]
        doc_labels = doc_labels[:128]
        # print(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)], doc_labels)
        counter = 0
        for i, (start, end) in enumerate(arr_offset):
            curr_token = encodings.encodings[enc_count].tokens[i]
            if start == 0 and end != 0 and curr_token != "<unk>" and "▁" in curr_token:
                doc_enc_labels[i] = doc_labels[counter]
                counter += 1
            elif (start != 0 and end != 0) or (end != 0 and curr_token not in ["<unk>", "<s>", "</s>", "<pad>"]):
                doc_enc_labels[i] = doc_enc_labels[max(i - 1, 0)]
        # if original_len > len(doc_labels):
        #     doc_labels.extend([tag2id["O"]] * (original_len-len(doc_labels)))
        # elif len(doc_labels) > original_len:
        #     doc_labels = doc_labels[:original_len]
        # set labels whose first offset position is 0 and the second is not 0
        # doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
        enc_count += 1

    return encoded_labels
