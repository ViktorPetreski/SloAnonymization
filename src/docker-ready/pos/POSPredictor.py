import logging
import string
import sys
from collections import defaultdict
from transformers import CamembertTokenizer, pipeline, CamembertForTokenClassification

class POSPredictor:
    def __init__(self, tokenizer_path_name, model_path_name):
        self.tokenizer_path = tokenizer_path_name
        self.model_path = model_path_name
        self.tag_dist = defaultdict(list)
        self.word_to_tag = defaultdict(str)
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrainL1OStrategy')
        self.nlp = None


    def init_tokenizer(self):
        self.logger.info("Init pos tokenizer")
        tokenizer = CamembertTokenizer.from_pretrained(
            self.tokenizer_path,
            from_pt=True,
            do_lower_case=False,
            local_files_only=False
        )
        return tokenizer

    def init_model(self):
        self.logger.info("Init pos model")
        return CamembertForTokenClassification.from_pretrained(self.model_path, local_files_only=True)

    def init_all(self):
        model = self.init_model()
        tokenizer = self.init_tokenizer()
        self.nlp = pipeline("token-classification",
                       model=model,
                       tokenizer=tokenizer,
                       ignore_labels=[""])

    def predict(self, text):
        self.logger.info("Start pos prediction")

        new_tokens = []
        new_labels = []
        nes = []
        final_labels = []
        tokens = self.nlp(text)
        word_token = []
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
            word_token.append(f"{word} [{label}]")
            self.tag_dist[label.lower()].append(word)
            key = word.lower().translate(str.maketrans('', '', string.punctuation))
            self.word_to_tag[key] = label.lower()
        self.logger.info("Finished predicting pos")
        return " ".join(word_token)

