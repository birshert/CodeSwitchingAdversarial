import os

import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset


def create_mapping(df):
    slot2idx = {t: i for i, t in enumerate({x for _ in df['slot_labels'].str.split().values for x in _})}
    slot2idx['PAD'] = len(slot2idx)
    slot2idx['UNK'] = len(slot2idx)

    idx2slot = {value: key for key, value in slot2idx.items()}

    intent2idx = {t: i for i, t in enumerate(df['intent'].unique())}
    intent2idx['UNK'] = len(intent2idx)

    return slot2idx, idx2slot, intent2idx


def prepare_datasets(tokenizer):
    if os.path.exists('data/cached'):
        train_dataset = torch.load('data/cached/train.pt')
        test_dataset = torch.load('data/cached/test.pt')
        num_slots, num_intents, idx2slot = torch.load('data/cached/misc.pt')

        return train_dataset, test_dataset, num_slots, num_intents, idx2slot

    train = pd.DataFrame()

    for language in ['EN', 'DE', 'ES', 'FR', 'JA', 'PT', 'ZH']:
        df = pd.read_csv(
            f'data/atis/train/train_{language}.tsv',
            delimiter='\t',
            index_col='id'
        )
        df['language'] = language
        train = pd.concat((train, df))

    train.reset_index(drop=True, inplace=True)

    test = pd.DataFrame()

    for language in ['EN', 'DE', 'ES', 'FR', 'JA', 'PT', 'ZH']:
        df = pd.read_csv(
            f'data/atis/test/test_{language}.tsv',
            delimiter='\t',
            index_col='id'
        )
        df['language'] = language
        test = pd.concat((test, df))

    test.reset_index(drop=True, inplace=True)

    slot2idx, idx2slot, intent2idx = create_mapping(train)
    num_slots, num_intents = len(slot2idx), len(intent2idx)

    def tokenize_and_preserve_labels(sentence, text_labels):
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

    train_data = []

    for index, row in train.iterrows():
        tokens, slot_labels = tokenize_and_preserve_labels(row['utterance'], row['slot_labels'])
        train_data.append((tokens, slot_labels, intent2idx.get(row['intent'], intent2idx['UNK'])))

    test_data = []

    for index, row in test.iterrows():
        tokens, slot_labels = tokenize_and_preserve_labels(row['utterance'], row['slot_labels'])
        test_data.append((tokens, slot_labels, intent2idx.get(row['intent'], intent2idx['UNK'])))

    def data2tensors(data):
        tokenized_texts = [elem[0] for elem in data]
        labels = [elem[1] for elem in data]
        intents = [elem[2] for elem in data]

        input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
            dtype='long',
            value=0.0,
            truncating='post',
            padding='post'
        )

        tags = pad_sequences(
            labels,
            value=slot2idx['PAD'],
            padding='post',
            dtype='long',
            truncating='post'
        )

        attention_masks = [[int(i != 0.0) for i in ii] for ii in input_ids]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(tags, dtype=torch.long),
            torch.tensor(intents, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
        )

    train_data = data2tensors(train_data)
    test_data = data2tensors(test_data)

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)

    os.mkdir('data/cached')
    torch.save(train_dataset, f'data/cached/train.pt')
    torch.save(test_dataset, f'data/cached/test.pt')
    torch.save((num_slots, num_intents, idx2slot), 'data/cached/misc.pt')

    return train_dataset, test_dataset, num_slots, num_intents, idx2slot
