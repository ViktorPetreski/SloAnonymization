import logging
import sys
from collections import defaultdict
from transformers import CamembertTokenizer, pipeline, CamembertForTokenClassification
from src.vp_ta_constants import TAG_SCHEME_COMBINED

class NERPredictor:
    def __init__(self, text, tokenizer_path_name, model_path_name) -> None:
        self.text = text
        self.tokenizer_path = tokenizer_path_name
        self.model_path = model_path_name
        self.tags = TAG_SCHEME_COMBINED
        self.tag_dist = defaultdict(list)
        self.word_to_tag = defaultdict(str)
        self.tagged_sentence = ""
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrainL1OStrategy')

    def init_tokenizer(self):
        self.logger.info("Init ner tokenizer")
        tokenizer = CamembertTokenizer.from_pretrained(
            self.tokenizer_path,
            from_pt=True,
            do_lower_case=False,
        )
        return tokenizer

    def init_model(self):
        self.logger.info("Init ner model")
        return CamembertForTokenClassification.from_pretrained(
            self.model_path, local_files_only=True)

    def predict(self):
        self.logger.info("Starting to predict ner")
        tag2id = {tag: str(id) for id, tag in enumerate(self.tags)}
        id2tag = {int(id): tag for tag, id in tag2id.items()}
        model = self.init_model()
        tokenizer = self.init_tokenizer()
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, ignore_labels=["O"])
        new_tokens = []
        new_labels = []
        nes = []
        final_labels = []
        word_token  = []
        tokens = nlp(self.text)
        # print(tokens)
        for tt in tokens:
            token = tt["word"]
            label = tt["entity"]
            if not token.startswith("▁"):
                new_tokens[-1] = new_tokens[-1] + token
            else:
                new_tokens.append(token[1:])
                new_labels.append(label)

        for label, token in zip(new_labels, new_tokens):
            if label.startswith("I"):
                nes[-1] = f"{nes[-1]} {token.lower()}"
            else:
                final_labels.append(label)
                nes.append(token.lower())

        for label, word in zip(final_labels, nes):
            self.tag_dist[label.replace("B-", "").lower()].append(word)
            word_token.append(f"{word} [{label}]")
            self.word_to_tag[word.lower()] = label.replace("B-", "").lower()
        self.tagged_sentence = " ".join(word_token)
        self.logger.info("Finished predicting ner")
        return self.tagged_sentence