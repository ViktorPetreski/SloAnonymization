# from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
# from src.coref.ContextualBertModel import ContextualControllerBERT
# from src.coref.data import Document
import json

from src.coref.ContextualBertModel import ContextualControllerBERT
from src.coref.data import Token, Mention, Document
import classla

def classla_output_to_coref_input(classla_output):
    # Transforms CLASSLA's output into a form that can be fed into coref model.
    output_tokens = {}
    output_sentences = []
    output_mentions = {}
    output_clusters = []

    str_document = classla_output.text
    start_char = 0
    MENTION_MSD = {"N", "V", "R", "P"}  # noun, verb, adverb, pronoun

    current_mention_id = 1
    token_index_in_document = 0
    for sentence_index, input_sentence in enumerate(classla_output.sentences):
        output_sentence = []
        mention_tokens = []
        for token_index_in_sentence, input_token in enumerate(input_sentence.tokens):
            input_word = input_token.words[0]
            output_token = Token(str(sentence_index) + "-" + str(token_index_in_sentence),
                                 input_word.text,
                                 input_word.lemma,
                                 input_word.xpos,
                                 sentence_index,
                                 token_index_in_sentence,
                                 token_index_in_document)

            # FIXME: This is a possibly inefficient way of finding start_char of a word. Stanza has this functionality
            #  implemented, Classla unfortunately does not, so we resort to a hack
            new_start_char = str_document.find(input_word.text, start_char)
            output_token.start_char = new_start_char
            if new_start_char != -1:
                start_char = new_start_char

            if len(mention_tokens) > 0 and mention_tokens[0].msd[0] != output_token.msd[0]:
                output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
                output_clusters.append([current_mention_id])
                mention_tokens = []
                current_mention_id += 1

            # Simplistic mention detection: consider nouns, verbs, adverbs and pronouns as mentions
            if output_token.msd[0] in MENTION_MSD:
                mention_tokens.append(output_token)

            output_tokens[output_token.token_id] = output_token
            output_sentence.append(output_token.token_id)
            token_index_in_document += 1

        # Handle possible leftover mention tokens at end of sentence
        if len(mention_tokens) > 0:
            output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
            output_clusters.append([current_mention_id])
            mention_tokens = []
            current_mention_id += 1

        output_sentences.append(output_sentence)

    return Document(1, output_tokens, output_sentences, output_mentions, output_clusters)


def init_classla():

    processors = 'tokenize,pos,lemma,ner'
    classla.download('sl', processors=processors)
    return classla.Pipeline('sl', processors=processors)

def init_coref():
    model =  ContextualControllerBERT.from_pretrained("../data_parser/contextual_model_bert/sloberta_TESTT_new")
    model.eval_mode()
    return model

if __name__ == '__main__':
    classla = init_classla()
    coref = init_coref()
    # 1. process input text with CLASSLA
    classla_output = classla("Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer.")

    # 2. re-format classla_output into coref_input (incl. mention detection)
    coref_input = classla_output_to_coref_input(classla_output)

    # 3. process prepared input with coref
    coref_output = coref.evaluate_single(coref_input)

    # 4. prepare response (mentions + coreferences)
    # 4. prepare response (mentions + coreferences)
    coreferences = []
    coreferenced_mentions = set()
    for id2, id1s in coref_output["predictions"].items():
        if id2 is not None:
            for id1 in id1s:
                mention_score = coref_output["scores"][id1]

                if mention_score < 0.3:
                    continue

                coreferenced_mentions.add(id1)
                coreferenced_mentions.add(id2)

                coreferences.append({
                    "id1": int(id1),
                    "id2": int(id2),
                    "score": mention_score
                })

    mentions = []
    for mention in coref_input.mentions.values():
        [sentence_id, token_id] = [int(idx) for idx in mention.tokens[0].token_id.split("-")]
        mention_score = coref_output["scores"][mention.mention_id]



        if mention_score < 0.3:
            # while this is technically already filtered with coreferenced_mentions, singleton mentions aren't, but they
            # have some score too that can be thresholded.
            continue

        mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
        mentions.append(
            {
                "id": mention.mention_id,
                "start_idx": mention.tokens[0].start_char,
                "length": len(mention_raw_text),
                # "ner_type": classla_output.sentences[sentence_id].tokens[token_id].ner,
                # "msd": mention.tokens[0].msd,
                "text": mention_raw_text
            }
        )

    print( {
        "mentions": mentions,
        "coreferences": sorted(coreferences, key=lambda x: x["id1"])
    })
