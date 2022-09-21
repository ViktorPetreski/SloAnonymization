# from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
# from src.coref.ContextualBertModel import ContextualControllerBERT
# from src.coref.data import Document
import json
import logging
import sys
from allennlp.predictors.predictor import Predictor

class CorefResolver:

    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.corefs: dict = {}
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrainL1OStrategy')
        self.logger.info("Initialized")
        self.predictor = Predictor.from_path(self.model_path)
        # self.classla = classla_nlp


    def get_span_words(self, span, document):
        return ' '.join(document[span[0]:span[1] + 1])

    def print_clusters(self, prediction):
        document, clusters = prediction['document'], prediction['clusters']
        for cluster in clusters:
            print(self.get_span_words(cluster[0], document) + ': ', end='')
            print(f"[{'; '.join([self.get_span_words(span, document) for span in cluster])}]")

    def resolve_allennlp(self, text):
        self.logger.info("Init allennlp predictor")

        self.logger.info("Start allennlp resolvee")
        prediction = self.predictor.predict(document=text)
        self.logger.info("Finish allennlp resolve")
        print(self.print_clusters(prediction))
        return self.predictor.coref_resolved(text)


