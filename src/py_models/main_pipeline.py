# from src.predictor.predict_ner import predict_ner
# from transformers import CamembertTokenizer, pipeline, CamembertForTokenClassification
# from pathlib import Path
# import json
# # from src.vp_ta_constants import TAG_SCHEME_COMBINED
# import pandas as pd
from src.predictor.predict_ner import predict_ner




if __name__ == '__main__':
    sentences = [
        "Trg prijateljstva je bil zaprt in zastražen s policisti, otroke iz osnovne šole in vrtca pa so po poročanju spletnega portala Kamnik.info evakuirali v Dom kulture Kamnik.",
        "Slovenija bo Avstraliji odprodala 20.400 odmerkov cepiva proti covidu-19 proizvajalca Moderna, je na današnji dopisni seji odločila slovenska vlada."
    ]

    predict_ner(sentences)
