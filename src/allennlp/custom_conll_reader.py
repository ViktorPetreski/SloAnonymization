import logging
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp_models.common.ontonotes import Ontonotes
from allennlp_models.coref.util import make_coref_instance

logger = logging.getLogger(__name__)

from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import os
import logging

from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedSpan
from nltk import Tree

logger = logging.getLogger(__name__)


class OntonotesSentence:
    """
    A class representing the annotations available for a single CONLL formatted sentence.

    # Parameters

    document_id : `str`
        This is a variation on the document filename
    sentence_id : `int`
        The integer ID of the sentence within a document.
    words : `List[str]`
        This is the tokens as segmented/tokenized in the Treebank.
    pos_tags : `List[str]`
        This is the Penn-Treebank-style part of speech. When parse information is missing,
        all parts of speech except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    parse_tree : `nltk.Tree`
        An nltk Tree representing the parse. It includes POS tags as pre-terminal nodes.
        When the parse information is missing, the parse will be `None`.
    predicate_lemmas : `List[Optional[str]]`
        The predicate lemma of the words for which we have semantic role
        information or word sense information. All other indices are `None`.
    predicate_framenet_ids : `List[Optional[int]]`
        The PropBank frameset ID of the lemmas in `predicate_lemmas`, or `None`.
    word_senses : `List[Optional[float]]`
        The word senses for the words in the sentence, or `None`. These are floats
        because the word sense can have values after the decimal, like `1.1`.
    speakers : `List[Optional[str]]`
        The speaker information for the words in the sentence, if present, or `None`
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    named_entities : `List[str]`
        The BIO tags for named entities in the sentence.
    srl_frames : `List[Tuple[str, List[str]]]`
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
    coref_spans : `Set[TypedSpan]`
        The spans for entity mentions involved in coreference resolution within the sentence.
        Each element is a tuple composed of (cluster_id, (start_index, end_index)). Indices
        are `inclusive`.
    """

    def __init__(
            self,
            document_id: str,
            sentence_id: int,
            words: List[str],
            pos_tags: List[str],
            parse_tree: Optional[Tree],
            predicate_lemmas: List[Optional[str]],
            named_entities: List[str],
            # srl_frames: List[Tuple[str, List[str]]],
            coref_spans: Set[TypedSpan],
    ) -> None:

        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.parse_tree = parse_tree
        self.predicate_lemmas = predicate_lemmas
        self.named_entities = named_entities
        # self.srl_frames = srl_frames
        self.coref_spans = coref_spans


class SentiCoref:

    def dataset_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> Iterator[List[OntonotesSentence]]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with codecs.open(file_path, "r", encoding="utf8") as open_file:
            conll_rows = []
            document: List[OntonotesSentence] = []
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(self._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    def sentence_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for document in self.dataset_document_iterator(file_path):
            for sentence in document:
                yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:
        document_id: str = None
        sentence_id: int = None

        # The words in the sentence.
        sentence: List[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: List[str] = []
        # the pieces of the parse tree.
        parse_pieces: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: List[str] = []
        # The FrameNet ID of the predicate.
        predicate_framenet_ids: List[str] = []
        # The sense of the word, if available.
        word_senses: List[float] = []
        # The current speaker, if available.
        speakers: List[str] = []

        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        # Cluster id -> List of (start_index, end_index) spans.
        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        # Cluster id -> List of start_indices which are open for this id.
        coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)

        for index, row in enumerate(conll_rows):
            conll_components = row.split()
            # print(document_id)
            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]
            # parse_piece = conll_components[5]

            # Replace brackets in text and pos tags
            # with a different token for parse trees.
            if pos_tag != "XX" and word != "XX":
                if word == "(":
                    parse_word = "-LRB-"
                elif word == ")":
                    parse_word = "-RRB-"
                else:
                    parse_word = word
                if pos_tag == "(":
                    pos_tag = "-LRB-"
                if pos_tag == ")":
                    pos_tag = "-RRB-"
                # (left_brackets, right_hand_side) = parse_piece.split("*")
                # only keep ')' if there are nested brackets with nothing in them.
                # right_brackets = right_hand_side.count(")") * ")"
                # parse_piece = f"{left_brackets} ({pos_tag} {parse_word}) {right_brackets}"
            else:
                # There are some bad annotations in the CONLL data.
                # They contain no information, so to make this explicit,
                # we just set the parse piece to be None which will result
                # in the overall parse tree being None.
                parse_piece = None

            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[10:-1]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:-1]]

            self._process_span_annotations_for_word(
                conll_components[10:-1], span_labels, current_span_labels
            )

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            # word_is_verbal_predicate = any("(V" in x for x in conll_components[11:-1])
            # if word_is_verbal_predicate:
            #     verbal_predicates.append(word)

            self._process_coref_span_annotations_for_word(
                conll_components[-1], index, clusters, coref_stacks
            )

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append("")
            predicate_lemmas.append(lemmatised_word if lemmatised_word != "-" else None)
            predicate_framenet_ids.append(framenet_id if framenet_id != "-" else None)
            word_senses.append(float(word_sense) if word_sense != "-" else None)
            speakers.append(speaker if speaker != "-" else None)

        named_entities = span_labels[0]
        # srl_frames = [
        #     (predicate, labels) for predicate, labels in zip(verbal_predicates, span_labels[1:])
        # ]

        # if all(parse_pieces):
        #     parse_tree = Tree.fromstring("".join(parse_pieces))
        # else:
        parse_tree = None
        coref_span_tuples: Set[TypedSpan] = {
            (cluster_id, span) for cluster_id, span_list in clusters.items() for span in span_list
        }
        return OntonotesSentence(
            document_id,
            sentence_id,
            sentence,
            pos_tags,
            parse_tree,
            predicate_lemmas,
            named_entities,
            # srl_frames,
            coref_span_tuples,
        )

    @staticmethod
    def _process_coref_span_annotations_for_word(
            label: str,
            word_index: int,
            clusters: DefaultDict[int, List[Tuple[int, int]]],
            coref_stacks: DefaultDict[int, List[int]],
    ) -> None:
        if label != "-":
            for segment in label.strip().split("|"):
                # The conll representation of coref spans allows spans to
                # overlap. If spans end or begin at the same word, they are
                # separated by a "|".
                if segment[0] == "(":
                    # The span begins at this word.
                    if segment[-1] == ")":
                        # The span begins and ends at this word (single word span).
                        cluster_id = int(segment[1:-1])
                        clusters[cluster_id].append((word_index, word_index))
                    else:
                        # The span is starting, so we record the index of the word.
                        cluster_id = int(segment[1:])
                        coref_stacks[cluster_id].append(word_index)
                else:
                    # The span for this id is ending, but didn't start at this word.
                    # Retrieve the start index from the document state and
                    # add the span to the clusters for this id.
                    cluster_id = int(segment[:-1])
                    start = coref_stacks[cluster_id].pop()
                    clusters[cluster_id].append((start, word_index))

    @staticmethod
    def _process_span_annotations_for_word(
            annotations: List[str],
            span_labels: List[List[str]],
            current_span_labels: List[Optional[str]],
    ) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.

        # Parameters

        annotations : `List[str]`
            A list of labels to compute BIO tags for.
        span_labels : `List[List[str]]`
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : `List[Optional[str]]`
            The currently open span per annotation type, or `None` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None

@DatasetReader.register("senticoref")
class SentiCorefReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = `None`)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: `int`, optional (default = `None`)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = `False`)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them. Ontonotes shouldn't have these, and this option should be used for
        testing only.
    """

    def __init__(
            self,
            max_span_width: int,
            token_indexers: Dict[str, TokenIndexer] = None,
            wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
            max_sentences: int = None,
            remove_singleton_clusters: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer
        self._max_sentences = max_sentences
        self._remove_singleton_clusters = remove_singleton_clusters

    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = SentiCoref()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens, end + total_tokens))
                total_tokens += len(sentence.words)

            yield self.text_to_instance([s.words for s in sentences], list(clusters.values()))

    def text_to_instance(
            self,  # type: ignore
            sentences: List[List[str]],
            gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Instance:
        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            gold_clusters,
            self._wordpiece_modeling_tokenizer,
            self._max_sentences,
            self._remove_singleton_clusters,
        )