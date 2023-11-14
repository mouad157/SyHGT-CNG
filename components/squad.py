# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import jsonpickle
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from components.hg_parser import ConstituencyParser, ConstituencyNode

from transformers.data.processors.utils import DataProcessor
from transformers.utils import is_tf_available, is_torch_available, logging
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from torch_geometric.data import HeteroData
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.nn import to_hetero
import torch_geometric
# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

l=False
if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset, Dataset

logger = logging.get_logger(__name__)
dicto = {"CD":0,
        "CC"  : 1,
        "DT" : 2,
        "EX" : 3,
        "FW" : 4,
        "IN" : 5,
        "JJ" : 6,
        "JJR" : 7,
        "JJS" : 8,
        "LS" : 9,
        "MD" : 10,
        "NN" : 11,
        "NNS" : 12,
        "NNP" : 13,
        "NNPS" : 14,
        "PDT" : 15,
        "POS" : 16,
        "PRP" : 17,
        "PP$" : 18,
        "RB" : 19,
        "RBR" : 20,
        "RBS" : 21,
        "RP" : 22,
        "SYM" : 23,
        "TO" : 24,
        "UH" : 25,
        "VB" : 26,
        "VBD" : 27,
        "VBG" : 28,
        "VBN"  : 29,
        "VBP" : 30,
        "VBZ" : 31,
        "WDT" : 32,
        "WP" : 33,
        "WP$" : 34,
        "WRB" : 35,
        "#" : 36,
        "." : 37,
        "$" : 38,
        "," : 39,
        ":" : 40,
        "(" : 41,
        ")" : 42,
        "`" : 43,
        "ROOT" : 44,
        "ADJP" : 45, 
        "ADVP" : 46, 
        "NP" : 47, 
        "PP" : 48, 
        "S" : 49, 
        "SBAR" : 50, 
        "SBARQ" : 51, 
        "SINV" :52 , 
        "SQ" :53 , 
        "VP" :54 , 
        "WHADVP" : 55, 
        "WHNP" : 56, 
        "WHPP" :57 , 
        "X" : 58,
        "*" : 59, 
        "0" : 59, 
        "T" : 59, 
        "NIL" :59 , 
        "PRT":60,
        "PRT$": 61,
        "PRP$": 62,
        "HYPH": 63,
        "NML":64,
        "AFX":65,
        "-LRB-":66,
        "CONJP" : 67,
        "-RRB-" : 68,
        "``": 69,
        "INTJ" : 70,
        "QP" : 71,
        "''" : 72,
        "WHADJP" : 73,
        "PRN" : 74,
        "UCP" : 75,
        "ADD" : 76,
        "FRAG" : 77,
        "NFP"  : 78,
        "RRC" : 79,
        'LST' : 80
}

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training):
    features = []
    if is_training and  not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        #print(cleaned_answer_text)
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []

    if not is_training and not example.is_impossible:
        start_position = example.start_position
        end_position = example.end_position
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        #print(cleaned_answer_text)
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []


    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        
        tok_start_position = orig_to_tok_index[example.start_position]
        # print(example.start_position)
        # print(tok_start_position)
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )
    if not is_training and not example.is_impossible:
        
        tok_start_position = orig_to_tok_index[example.start_position]
        # print(example.start_position)
        # print(tok_start_position)
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

        #print(all_doc_tokens[tok_start_position],all_doc_tokens[tok_end_position])
    spans = []
    example.question_text = example.question_text.replace("£", "£ ")
    example.question_text= example.question_text.replace("$", "$ ")
    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value
        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        sep_index = span["input_ids"].index(tokenizer.sep_token_id)

        # p_mask: mask with 1 for token that cannot be in the answer (0 for token which can be in an answer)
        # Original TF implementation also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if   not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        
        # print("input_ids", span["input_ids"])
        # print("tokens", span["tokens"])
        # print("token_to_orig_map", span["token_to_orig_map"])
        # print("start", span["start"])
        

        # Token node feature mapping dict
        token_node_mapping = dict()
        docto = dict()
        for index, token in enumerate(span_doc_tokens):
            docto[token] = len(truncated_query) + sequence_added_tokens + index
            token_node_mapping[index] = len(truncated_query) + sequence_added_tokens + index
        #print(token_node_mapping)
        # print(docto)
        # print(start_position,end_position)
        #print()
        #print(span_doc_tokens[start_position],span_doc_tokens[end_position])
        #Leaf node feature mapping dict
        leaf_id = 0
        last_token_id = sep_index + 1
        leaf_node_mapping = dict()

        span["context"] = tokenizer.decode(span["input_ids"][sep_index+1:], skip_special_tokens=True)
        # c_parser = ConstituencyParser(use_gpu=False)

        # constituents = c_parser.get_sentences(" ".join(span["context"].split()))
        # print(constituents)

        # for constituent in constituents:
        #     leaves = constituent.constituency.leaf_labels()
        #     for leaf in leaves:
        #         leaf = leaf.lower()
        #         current_token_text = ""
        #         current_token_ids = list()
        #         while current_token_text != leaf:
        #             current_token_text += span["tokens"][last_token_id] if not span["tokens"][last_token_id].startswith("##") else span["tokens"][last_token_id][2:]
        #             current_token_ids += [last_token_id]
        #             last_token_id += 1
        #         leaf_node_mapping[leaf_id] = current_token_ids
    #         leaf_id += 1
        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                sep_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                doc_tokens = span_doc_tokens,
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of [`~data.processors.squad.SquadExample`]
        tokenizer: an instance of a child of [`PreTrainedTokenizer`]
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.


    Returns:
        list of [`~data.processors.squad.SquadFeatures`]

    Example:

    ```python
    processor = SquadV2Processor()
    examples = processor.get_dev_examples(data_dir)

    features = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
    )
    ```"""
    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=16),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    # Add example index and unique id
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Constituency Graph Construction
    # Stanford NLP Parser doesn‘t support multiprocessing mode, so that we need to operate it with the single track.
    c_parser = ConstituencyParser()
    graph_f = open("./data/squad_v2/graph/v2_test.json", "w")
    graph_ids = list()
    graph_id = 0
    error = 0
    error2 = 0
    for feature in tqdm(features, desc="constituency graph construction"):
        #print(feature.__dict__.keys())
        # 0) Meta info
        start_position = feature.start_position
        toki = feature.tokens
        end_position = feature.end_position
        qas_id = feature.qas_id
        context = tokenizer.decode(feature.input_ids, skip_special_tokens=True) #, clean_up_tokenization_spaces=False)
        
        toki.remove("[CLS]")
        toki.remove("[SEP]")

        # print(feature.input_ids)
        #print(start_position)
        #print(end_position)
        # print(feature.input_ids)
        # print(len(feature.input_ids))
        # print(tokenizer.decode(feature.input_ids, skip_special_tokens=True))
        # print("***************************************")
        # print(feature.input_ids[feature.sep_index+1:])
        # print(len(feature.input_ids[feature.sep_index+1:]))
        # print(context)
        # print("***************************************")
        #print(context)
        constituents = c_parser.get_sentences(context)
        #print(constituents)
        # 1) token2token feature mapping dict
        token2token_mapping = dict()
        # print(feature.doc_tokens)
        # print(feature.sep_index)
        for index, token in enumerate(feature.tokens):
            #print(feature.sep_index)
            token2token_mapping[index] =  index + feature.sep_index

        # 2) leaf2token feature mapping dict
        
        try:
            leaf_id = 0
            last_token_id =  0 #feature.sep_index +1
            leaf_node_mapping = dict()
            text_node_mapping = dict()
            for constituent in constituents:
                leaves = constituent.constituency.leaf_labels()
                #print(leaves)
                for leaf in leaves:
                    leaf = leaf.lower()
                    current_token_text = ""
                    current_token_ids = list()
                    while current_token_text != leaf:
                        current_token_text += toki[last_token_id] if not toki[last_token_id].startswith("##") else toki[last_token_id][2:]
                        current_token_ids += [last_token_id]
                        last_token_id += 1
                    leaf_node_mapping[leaf_id] = current_token_ids
                    text_node_mapping[leaf_id] = current_token_text
                    leaf_id += 1
            #print("leaf_node_mapping",leaf_node_mapping)
        except Exception as e:
            # If the preterminals are different with tokens
            print("Constituents Leaf Parser Failed, Skip this one: {e}")
            print(leaves)
            print(feature.tokens[feature.sep_index +1:])
            error =error+1
            graph_ids +=[-1]
            continue

        # 3) constituent2leaf feature mapping dict
        
        try:
            pid,cid  = 0,len(leaf_node_mapping)
            #R_cc, R_ct = list(), list()
            def iterate_tree(root,R_cc,R_ct,labels):
                nonlocal pid,cid
                if root.is_preterminal():

                    tids=leaf_node_mapping[pid]
                    is_answer = False
                    if tids[0] == start_position-2 and tids[-1] ==  end_position-2 and end_position != 0:
                        is_answer = True
                    leaf_node = ConstituencyNode(cid=pid, label=root.label, text=root.leaf_labels(), lids=[pid], tids=tids, children=[], is_answer=is_answer)
                    R_ct.extend([[pid, tid] for tid in tids])
                    labels[pid] = (root.label,is_answer)
                    pid =pid + 1
                    return leaf_node , R_cc , R_ct, labels
                else:
                    child_nodes, lids, tids = list(), list(), list()
                    labels1 = dict()
                    for child in root.children:
                        child_node, R_cc, R_ct, labe = iterate_tree(child,R_cc,R_ct,labels)
                        labels1.update(labe)

                        #print("**")
                        child_nodes += [child_node]
                        lids += child_node.lids
                        for lid in child_node.lids:
                            tids += leaf_node_mapping[lid]
                    is_answer = False
                    tids.sort()
                    if tids[0] ==  start_position-2 and tids[-1] == end_position-2 and end_position != 0:
                        # print(toki[start_position], toki[end_position])
                        is_answer = True
                    leaf_node = ConstituencyNode(cid=cid, label=root.label, text=root.leaf_labels(), lids=lids, tids=tids, children=child_nodes, is_answer=is_answer)
                    R_cc = R_cc + [[cid, cild.cid] for cild in child_nodes]
                    labels1[cid] = (root.label,is_answer)
                    #R_ct = R_ct +  [[pid, tid] for tid in tids]
                    cid = cid + 1
                    return leaf_node ,R_cc, R_ct, labels1
            
            child_nodes, lids, tids = list(), list(), list()
            labels=dict()
            #print(R_ct)
            #print(constituents)
            r_t = list()
            r_c = list()
            for constituent in constituents:
                #print(type(constituent))
                try:
                    root, R_cc, R_ct, lab = iterate_tree(constituent.constituency,list(),list(),dict())
                    r_t.extend(R_ct)
                    r_c.extend(R_cc)
                    labels.update(lab)
                    child_nodes += [root]
                    lids += root.lids
                    for lid in root.lids:
                        tids += leaf_node_mapping[lid]
                except:
                    # print(constituent.constituency)
                    # print(root.text)
                    pass
            r_c = r_c + [[cid, cild.cid] for cild in child_nodes]
            labels.update({cid:("ROOT",False)})
            c_graph_node = ConstituencyNode(cid=cid, label="CONTEXT", text=context, lids=lids, tids=tids, children=child_nodes, is_answer=False)
            #print(labels)
            pre=[]
            pre2 = []
            #print(labels)
            for key in labels:
                sam = torch.zeros(82)
                lob = labels[key]
                number = dicto[lob[0]]
                if lob[1] == True:
                    pre2.append(1)
                else:
                    pre2.append(0)
                sam[number] = 1
                pre.append(sam)

            resu = torch.stack(pre,0)
            y = torch.tensor(pre2,dtype=torch.long)
            #print(y.size())
            #print(resu.size())
            #print("*")
            c_edge_index = {
                ("token", "connect", "constituent"): torch.tensor(r_t).contiguous(),
                ("constituent", "connect", "constituent"): torch.tensor(r_c).contiguous()
            }

            data = HeteroData()
            data["constituent"].node_id = torch.arange(len(labels))
            data["token"].node_id = torch.arange(len(tids))
            data["token"].x = torch.reshape(torch.tensor(feature.input_ids),(len(feature.input_ids),1))
            data["constituent"].x = resu
            data["constituent"].y = y

            data["constituent","connect","constituent"] = torch.tensor(r_c).contiguous()
            data["constituent","connect","token"] = torch.tensor(r_t).contiguous()
            graph_f.write(jsonpickle.encode({"qid": graph_id, "graph": data}, indent=None) + "\n")
            graph_ids += [graph_id]
            graph_id += 1

        except Exception as e:
            print(f"Constituents Token Parser Failed, Skip This One: {e}")
            #print(lab)
            #print(root.text)
            #print(labels)
            graph_ids += [-1]
            error2 = error2 +1
            continue

    graph_f.close()
    print(error)
    print(error2)
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_sep_index = torch.tensor([f.sep_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_qas_id = torch.tensor(graph_ids, dtype=torch.long)


        if not is_training:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, 
                all_attention_masks, 
                all_token_type_ids, 
                all_start_positions,
                all_end_positions,
                all_feature_index, 
                all_cls_index, 
                all_sep_index,
                all_p_mask,
                all_qas_id
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_sep_index,
                all_p_mask,
                all_is_impossible,
                all_qas_id
            )

        return features, dataset
    else:
        return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            # answer = None
            # answer_start = None
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of [`~data.processors.squad.SquadExample`] using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from *tensorflow_datasets.load("squad")*
            evaluate: Boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples:

        ```python
        >>> import tensorflow_datasets as tfds

        >>> dataset = tfds.load("squad")

        >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
        >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        ```"""

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset, desc="get examples from dataset"):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data, desc="create examples"):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]


                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        # constituents=constituents,
                    )
                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    [`~data.processors.squad.SquadExample`] using the
    :method:*~transformers.data.processors.squad.squad_convert_examples_to_features* method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context:
            List of booleans identifying which tokens have their maximum context in this feature object. If a token
            does not have their maximum context in this feature object, it means that another feature object has more
            information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignment methods.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        sep_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        doc_tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
        encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.sep_index = sep_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.doc_tokens = doc_tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.encoding = encoding


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
if __name__ == "__main__":
    # 1. Constituency Parser
    features = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training)