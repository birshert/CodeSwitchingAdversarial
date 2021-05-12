import logging
import re

import numpy as np
import torch
import yaml
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score

from model import JointMBERT
from model import JointXLMRoberta
from model import MLMMBERT
from model import MLMXLMRoberta


model_mapping = {
    'xlm-r': JointXLMRoberta,
    'm-bert': JointMBERT,
    'mlm xlm-r': MLMXLMRoberta,
    'mlm m-bert': MLMMBERT
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


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels, slot2idx):
    if isinstance(sentence, str):
        sentence = sentence.split()

    if isinstance(text_labels, str):
        text_labels = text_labels.split()

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([slot2idx.get(label, slot2idx['UNK'])] * len(tokenized_word))

    return tokenized_sentence, labels


def create_mapping(df):
    labels = ['PAD', 'UNK'] + list(sorted({x for _ in df['slot_labels'].str.split().values for x in _}))
    slot2idx = {t: i for i, t in enumerate(labels)}

    idx2slot = {value: key for key, value in slot2idx.items()}

    intent2idx = {t: i for i, t in enumerate(df['intent'].unique())}
    intent2idx['UNK'] = len(intent2idx)

    return slot2idx, idx2slot, intent2idx


def load_config(config_path: str = 'config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_config(config, config_path: str = 'config.yaml'):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, Dumper=yaml.SafeDumper)


def _set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
