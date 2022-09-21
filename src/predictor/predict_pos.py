# from transformers import CamembertTokenizer, pipeline, CamembertForTokenClassification
from pathlib import Path
import json
from src.vp_ta_constants import TAG_SCHEME_COMBINED, COUNTRIES, SLOVENIAN_MUNICIPALITIES, SLOVENIAN_CITIES
import pandas as pd
import gettext

def get_tags():
    tag_path = Path("../../results/xpos/custom_xpos_model_sloberta—combined_all—15e—32b—0ws—6e-05lr_new_test_new_test.csv")
    df = pd.read_csv(tag_path)
    return df["Level"].to_list()[:-2]

if __name__ == '__main__':
    # sentences = [
    #     "Trg prijateljstva je bil zaprt in zastražen s policisti, otroke iz osnovne šole in vrtca pa so po poročanju spletnega portala Kamnik.info evakuirali v Dom kulture Kamnik.",
    #     "Slovenski predsednik Borut Pahor je danes na Brdu pri Kranju priredil 11. vrh voditeljev procesa Brdo – Brioni.",
    #     "Edinstveno brezplačno rekreativno prireditev, skupinsko rolanje za tovornjakom na ulicah Maribora. Pričetek v nedeljo ob 15h na Trgu svobode.",
    #     "Mesečna parkirnina za abonente na javnih plačljivih parkiriščih: Rakušev trg, Sodna ulica 25, Mlinska ulica, Loška ulica in Parkirišče ULICA HEROJA BRAČIČA"
    #     "Vishka Cesta 43 pfokqpwfok powfkepwofqkepo +38669699034 fewfpekwf ofwekf poekwfpokew pfokpowe few +38669699034 fweofkpewofkpeof kpfoewkf poewkfpok pofewkfe +38669699043"
    # ]
    # # tags = get_tags()
    # tags = TAG_SCHEME_COMBINED
    # # tags.append("-PAD-")
    # #print(tags)
    # tokenizer_path = Path("../../models/sloberta-2.0")
    # model_path = Path("../../models/ner/custom_ner_model_sloberta—combined—5e—8b—500ws—5e-05lr_new_test")
    # tokenizer = CamembertTokenizer.from_pretrained(
    #     tokenizer_path,
    #     from_pt=True,
    #     do_lower_case=False,
    # )
    # tag2id = {tag: str(id) for id, tag in enumerate(tags)}
    # id2tag = {int(id): tag for tag, id in tag2id.items()}
    # model = CamembertForTokenClassification.from_pretrained(
    #     model_path,
    #     id2label=id2tag,
    #     label2id=tag2id,
    #     num_labels= len(tags)
    # )
    # nlp = pipeline("ner", model=model, tokenizer=tokenizer, ignore_labels=["O"])
    # for s in sentences:
    #     new_tokens = []
    #     nes = []
    #     new_labels = []
    #     final_labels = []
    #     tokens = nlp(s)
    #     # print(tokens)
    #     for tt in tokens:
    #         token = tt["word"]
    #         label = tt["entity"]
    #         if not token.startswith("▁"):
    #             new_tokens[-1] = new_tokens[-1] + token
    #         else:
    #             new_tokens.append(token[1:])
    #             new_labels.append(label)
    #
    #     for label, token in zip(new_labels, new_tokens):
    #         if label.startswith("I"):
    #             nes[-1] = f"{nes[-1]} {token.lower()}"
    #         else:
    #             final_labels.append(label)
    #             nes.append(token.lower())
    #
    #     for label, word in zip(final_labels, nes):
    #         print(f"{word} [{label}]", end=" ")
    #     print("\n")

    print([x.lower() for x in SLOVENIAN_CITIES])
