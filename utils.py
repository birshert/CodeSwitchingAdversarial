import logging
import re

import numpy as np
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
)

from model import XLMRoberta


MODEL_CLASSES = {
    'xlm-r': (XLMRobertaConfig, XLMRoberta, XLMRobertaTokenizerFast),
}

MODEL_PATH_MAP = {
    'xlm-r': 'xlm-roberta-base',
}


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    results = {}

    results.update(get_intent_acc(intent_preds, intent_labels))
    results.update(get_slot_metrics(slot_preds, slot_labels))
    results.update(get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels))

    return results


def get_slot_metrics(preds, labels):
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    return {
        "intent_acc": np.mean(preds == labels)
    }


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    slot_result = []

    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)

    return {
        "sementic_frame_acc": np.multiply((intent_preds == intent_labels), np.array(slot_result)).mean()
    }


def set_global_logging_level(level=logging.ERROR, prefices=None):
    if prefices is None:
        prefices = [""]
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
