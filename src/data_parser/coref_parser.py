import logging
import sys
from src.coref.ContextualBertModel import ContextualControllerBERT
from src.coref.data import read_corpus
from src.coref.utils import split_into_sets

if __name__ == '__main__':
    # doc = open_document("../../data/SentiCoref_1.0/1.tsv")
    # res = _prepare_doc(doc)
    # print(res)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    documents = read_corpus("senticoref")


    def create_model_instance(model_name, **override_kwargs):
            return ContextualControllerBERT(model_name=model_name,
                                            fc_hidden_size=64,
                                            dropout=0.1,
                                            combine_layers=True,
                                            pretrained_model_name_or_path="EMBEDIA/sloberta",
                                            learning_rate=6e-5,
                                            layer_learning_rate={"lr_embedder": 2e-5},
                                            max_segment_size=512,
                                            dataset_name="senticoref",
                                            freeze_pretrained=False)


    logging.info(f"Using single train/dev/test split...")
    # if args.fixed_split:
    #     logging.info("Using fixed dataset split")
    #     train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
    # else:
    train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

    model = create_model_instance(model_name="sloberta_TESTT_new")
    if not model.loaded_from_file:
        model.train(epochs=5, train_docs=train_docs, dev_docs=dev_docs)
        # Reload best checkpoint
        model = ContextualControllerBERT.from_pretrained(model.path_model_dir)

    model.evaluate(test_docs)
    # model.visualize()