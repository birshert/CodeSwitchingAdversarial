import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from model import BaseJointModel
from model import BaseMLMModel
from utils import create_mapping
from utils import load_config
from utils import tokenize_and_preserve_labels


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


class CustomJointDataset(Dataset):

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


class JointCollator:

    def __init__(self, slot2idx):
        self.slot2idx = slot2idx

    def __call__(self, x):
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


class CustomMLMDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = [tokenizer(txt, return_tensors='pt', return_attention_mask=False)['input_ids'][0] for txt in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MLMCollator:

    def __init__(self, tokenizer, mlm_probability):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, x):
        input_ids = pad_sequence(x, batch_first=True)
        input_ids, labels = self.mask_tokens(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    def mask_tokens(self, inputs):
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def prepare_joint_datasets(config, model: BaseJointModel):
    cached_path = f'data/{config["dataset"]}_cached_{model.__model_name__}'

    if os.path.exists(cached_path):
        train_dataset = torch.load(cached_path + '/train.pt')
        test_dataset = torch.load(cached_path + '/test.pt')
        collator, slot2idx, idx2slot = torch.load(cached_path + '/misc.pt')

        return train_dataset, test_dataset, collator, slot2idx, idx2slot

    train = read_atis('train')
    test = read_atis('test')

    slot2idx, idx2slot, intent2idx = create_mapping(train)

    train_data = []

    for index, row in train.iterrows():
        tokens, slot_labels = tokenize_and_preserve_labels(
            model.tokenizer,
            row['utterance'],
            row['slot_labels'],
            slot2idx
        )
        train_data.append(
            (
                tokens,
                slot_labels,
                intent2idx.get(row['intent'], intent2idx['UNK'])
            )
        )

    test_data = []

    for index, row in test.iterrows():
        tokens, slot_labels = tokenize_and_preserve_labels(
            model.tokenizer,
            row['utterance'],
            row['slot_labels'],
            slot2idx
        )
        test_data.append(
            (
                tokens,
                slot_labels,
                intent2idx.get(row['intent'], intent2idx['UNK'])
            )
        )

    train_dataset = CustomJointDataset(train_data, model.tokenizer, slot2idx)
    test_dataset = CustomJointDataset(test_data, model.tokenizer, slot2idx)
    collator = JointCollator(slot2idx)

    os.mkdir(cached_path)
    torch.save(train_dataset, cached_path + '/train.pt')
    torch.save(test_dataset, cached_path + '/test.pt')
    torch.save((collator, slot2idx, idx2slot), cached_path + '/misc.pt')

    return train_dataset, test_dataset, collator, slot2idx, idx2slot


def prepare_mlm_datasets(config, model: BaseMLMModel):
    cached_path = f'data/{config["dataset"]}_cached_{model.__model_name__}'

    if os.path.exists(cached_path):
        train_dataset = torch.load(cached_path + '/train.pt')
        test_dataset = torch.load(cached_path + '/test.pt')
        collator = torch.load(cached_path + '/misc.pt')

        return train_dataset, test_dataset, collator

    data = read_atis('adversarial', ['en', 'de'])
    uuids = data['uuid']

    uuids_train, uuids_test = train_test_split(uuids, test_size=0.1)

    train = data.loc[data['uuid'].isin(uuids_train)]['utterance']
    test = data.loc[data['uuid'].isin(uuids_test)]['utterance']

    train_dataset = CustomMLMDataset(train, model.tokenizer)
    test_dataset = CustomMLMDataset(test, model.tokenizer)
    collator = MLMCollator(model.tokenizer, config.get('mlm_probability', None))

    os.mkdir(cached_path)
    torch.save(train_dataset, cached_path + '/train.pt')
    torch.save(test_dataset, cached_path + '/test.pt')
    torch.save(collator, cached_path + '/misc.pt')

    return train_dataset, test_dataset, collator
