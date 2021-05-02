import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import (
    create_mapping,
    load_config,
    tokenize_and_preserve_labels,
)

from model import BaseModel


def read_atis(subset: str, languages: list = None):
    if languages is None:
        languages = load_config()['languages']

    result = pd.DataFrame()

    for language in languages:
        df = pd.read_csv(
            f'data/atis/{subset}/{subset}_{language}.csv',
            index_col=0
        )
        df['language'] = language
        df['uuid'] = np.arange(len(df))
        result = pd.concat((result, df))

    result.reset_index(drop=True, inplace=True)

    return result


class CustomDataset(Dataset):

    def __init__(self, data, tokenizer, slot2idx):
        self.input_ids = [
            torch.tensor(tokenizer.convert_tokens_to_ids(txt), dtype=torch.long) for txt in [elem[0] for elem in data]
        ]
        self.labels = [torch.tensor(elem[1], dtype=torch.long) for elem in data]
        self.intents = [torch.tensor(elem[2], dtype=torch.long) for elem in data]

        self.slot2idx = slot2idx

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return (
            self.input_ids[item],
            self.labels[item],
            self.intents[item]
        )

    def collate_fn(self, x):
        input_ids = [elem[0] for elem in x]
        labels = [elem[1] for elem in x]
        intents = [elem[2] for elem in x]

        input_ids = pad_sequence(input_ids, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.slot2idx['PAD'])

        return {
            'input_ids': input_ids.to(torch.long),
            'slot_labels_ids': labels.to(torch.long),
            'intent_label_ids': torch.tensor(intents, dtype=torch.long),
            'attention_mask': (input_ids != 0).to(torch.long)
        }


def prepare_datasets(model: BaseModel):
    cached_path = f'data/cached_{model.__model_name__}'

    if os.path.exists(cached_path):
        train_dataset = torch.load(cached_path + '/train.pt')
        test_dataset = torch.load(cached_path + '/test.pt')
        slot2idx, idx2slot = torch.load(cached_path + '/misc.pt')

        return train_dataset, test_dataset, slot2idx, idx2slot

    train = read_atis('train')
    test = read_atis('test')

    slot2idx, idx2slot, intent2idx = create_mapping(train)

    train_data = []

    for index, row in train.iterrows():
        tokens, slot_labels = tokenize_and_preserve_labels(model.tokenizer, row['utterance'], row['slot_labels'], slot2idx)
        train_data.append((tokens, slot_labels, intent2idx.get(row['intent'], intent2idx['UNK'])))

    test_data = []

    for index, row in test.iterrows():
        tokens, slot_labels = tokenize_and_preserve_labels(model.tokenizer, row['utterance'], row['slot_labels'], slot2idx)
        test_data.append((tokens, slot_labels, intent2idx.get(row['intent'], intent2idx['UNK'])))

    train_dataset = CustomDataset(train_data, model.tokenizer, slot2idx)
    test_dataset = CustomDataset(test_data, model.tokenizer, slot2idx)

    os.mkdir(cached_path)
    torch.save(train_dataset, cached_path + '/train.pt')
    torch.save(test_dataset, cached_path + '/test.pt')
    torch.save((slot2idx, idx2slot), cached_path + '/misc.pt')

    return train_dataset, test_dataset, slot2idx, idx2slot
