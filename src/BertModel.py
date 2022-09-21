import sys
from typing import List
from pathlib import Path
import pickle
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelForTokenClassification
from keras.preprocessing.sequence import pad_sequences


from src.ner import utils

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainL1OStrategy')


class BertModel:
    def __init__(
            self,
            metric: str,
            label_list: List[str],
            tag2id: dict[str: int],
            id2tag: dict[int: str],
            transformer_path: str,
            train: dict[str: List[str]],
            test: dict[str: List[str]],
            dev: dict[str: List[str]],
            task: str = "ner",
            dataset: str = "bsnlp",
            epochs:int = 3,
    ):
        self.input_model_path = transformer_path
        # self.metric = load_metric(metric)
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.label_list = label_list
        self.epochs = epochs
        self.MAX_LENGTH = 128  # max input length
        self.BATCH_SIZE = 32  # max input length
        self.task = task
        self.dataset = dataset
        self.train = train
        self.test = test
        self.dev = dev
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.input_model_path,
            from_pt=True,
            do_lower_case=False,
        )

    def make_datasets(self):
        train_labels = self.train["sentences"]
        train_tags = self.train["tags"]

        test_labels = self.test["sentences"]
        test_tags = self.test["tags"]

        dev_labels = self.dev["sentences"]
        dev_tags = self.dev["tags"]

        train_encodings_path = Path(f"../../data/encodings/{self.dataset}/{self.task}/train.pickle")
        test_encodings_path = Path(f"../../data/encodings/{self.dataset}/{self.task}/test.pickle")
        dev_encodings_path = Path(f"../../data/encodings/{self.dataset}/{self.task}/dev.pickle")

        if train_encodings_path.is_file() and test_encodings_path.is_file() and dev_encodings_path.is_file():
            train_encodings = pickle.load(open(train_encodings_path, "rb"))
            # train_encodings = train_encodings[:100]
            test_encodings = pickle.load(open(test_encodings_path, "rb"))
            dev_encodings = pickle.load(open(dev_encodings_path, "rb"))
            # dev_encodings = dev_encodings[:20]
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.input_model_path,
                from_pt=True,
                do_lower_case=False,
            )
            logger.info("Started generating encodings for train")
            train_encodings = utils.generate_encodings(tokenizer, train_labels)
            logger.info("Started generating encodings for test")
            test_encodings = utils.generate_encodings(tokenizer, test_labels)
            logger.info("Started generating encodings for dev")
            dev_encodings = utils.generate_encodings(tokenizer, dev_labels)

            pickle.dump(train_encodings, open(train_encodings_path, "wb"))
            pickle.dump(test_encodings, open(test_encodings_path, "wb"))
            pickle.dump(dev_encodings, open(dev_encodings_path, "wb"))

        train_labels = utils.encode_tags(train_tags, train_encodings, label_list)
        test_labels = utils.encode_tags(test_tags, test_encodings, label_list)
        dev_labels = utils.encode_tags(dev_tags, dev_encodings, label_list)

        train_dataset = utils.generate_dataset(train_encodings, train_labels)
        test_dataset = utils.generate_dataset(test_encodings, test_labels)
        val_dataset = utils.generate_dataset(dev_encodings, dev_labels)

        return train_dataset, test_dataset, val_dataset


    def convert_input(self, dataset: dict[str: List[str]]):
        tokens = []
        tags = []  # NER tags

        for tags, sentences in zip(dataset["tags"], dataset["sentences"]):
            sentence_tokens = []
            sentence_tags = []
            for tag, word in zip(tags, sentences):
                word_tokens = self.tokenizer.tokenize(str(word))
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([self.tag2id[tag]] * len(word_tokens))
            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            tokens.append(sentence_ids)
            tags.append(sentence_tags)

        tokens = torch.as_tensor(pad_sequences(
            tokens,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=-100.0,
            truncating="post",
            padding="post"
        )).to(self.device)
        tags = torch.as_tensor(pad_sequences(
            tags,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=self.tag2id["PAD"],
            truncating="post",
            padding="post"
        )).to(self.device)
        masks = torch.as_tensor(np.array([[float(token != 0.0) for token in sentence] for sentence in tokens])).to(
            self.device)
        data = TensorDataset(tokens, masks, tags)
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.BATCH_SIZE)

    def train(
            self,
            data_loaders: dict
    ):
        logger.info(f"Loading the pre-trained model `{self.input_model_path}`...")
        model = AutoModelForTokenClassification.from_pretrained(
            self.input_model_path,
            num_labels=len(self.tag2code),
            label2id=self.tag2id,
            id2label=self.id2tag,
            output_attentions=False,
            output_hidden_states=False
        )

        model = model.to(self.device)
        optimizer, loss = None, None

        for dataset, dataloader in data_loaders.items():
            logger.info(f'Training on `{dataset}`')
            # hack to use entire dataset, leaving the validation data intact
            td = pd.concat([dataloader.train(), dataloader.test()]) if self.use_test else dataloader.train()
            model, optimizer, loss = self.__train(model, train_data=td,
                                                  validation_data=dataloader.dev())

        out_fname = f"{self.output_model_path}/{self.output_model_fname}"
        logger.info(f"Saving the model at: {out_fname}")
        model.save_pretrained(out_fname)
        self.tokenizer.save_pretrained(out_fname)
        logger.info("Done!")

    def __train(
            self,
            model,
    ):
        logger.info("Loading the training data...")
        train_data = self.convert_input(self.train)
        logger.info("Loading the validation data...")
        validation_data = self.convert_input(self.dev)

        model_parameters = list(model.named_parameters())
        optimizer_parameters = [{"params": [p for n, p in model_parameters]}]

        optimizer = AdamW(
            optimizer_parameters,
            lr=3e-5,
            eps=1e-8
        )

        total_steps = len(train_data) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # ensure reproducibility
        # TODO: try out different seed values
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_loss, validation_loss, loss = [], [], None
        logger.info(f"Training the model for {self.epochs} epochs...")
        for _ in trange(self.epochs, desc="Epoch"):
            model.train()
            total_loss = 0
            # train:
            for step, batch in tqdm(enumerate(train_data), desc='Batch'):
                batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)

                # reset the grads
                model.zero_grad()

                outputs = model(
                    batch_tokens,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )

                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()

                # preventing exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_grad_norm)

                # update the parameters
                optimizer.step()

                # update the learning rate (lr)
                scheduler.step()

            avg_epoch_train_loss = total_loss / len(train_data)
            logger.info(f"Avg train loss = {avg_epoch_train_loss:.4f}")
            training_loss.append(avg_epoch_train_loss)

            # validate:
            model.eval()
            val_loss, val_acc, val_f1, val_p, val_r, val_report = self.__test(model, validation_data)
            validation_loss.append(val_loss)
            logger.info(f"Validation loss: {val_loss:.4f}")
            logger.info(f"Validation accuracy: {val_acc:.4f}, P: {val_p:.4f}, R: {val_r:.4f}, F1 score: {val_f1:.4f}")
            logger.info(f"Classification report:\n{val_report}")

        return model, optimizer, loss



