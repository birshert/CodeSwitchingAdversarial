import random

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from transformers import (
    XLMRobertaTokenizerFast,
)


class CustomDataloader(Dataset):
    """
    Returns tokenized sentence and masked sentence (currently masking random word from the sentence), using XLM-R
    tokenizer.
    """

    def __init__(self, sentences, shuffle: bool = True, batch_size: int = 100):
        self.sentences = [
            word_tokenize(sentence) for sentence in sentences
        ]
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

    def data(self):
        for item in range(len(self.sentences)):
            yield self.__getitem__(item)

    def __len__(self):
        raise NotImplementedError('Len cannot be calculated with not fixed batch size')

    def __getitem__(self, item: int):
        sentence = ' '.join(self.sentences[item])

        word_to_mask = np.random.choice(np.arange(0, len(self.sentences[item])))  # TODO: choose what words to mask

        masked_sentence = self.sentences[item]
        masked_sentence[word_to_mask] = self.tokenizer.mask_token
        masked_sentence = ' '.join(masked_sentence)

        origin = self.tokenizer.encode(sentence, return_tensors='pt').flatten()
        masked = self.tokenizer.encode(masked_sentence, return_tensors='pt').flatten()

        return {
            'origin': origin,
            'masked': masked,
            'origin_len': len(origin),
            'masked_len': len(masked)
        }

    @staticmethod
    def batch_size_fn(x):
        return x['origin_len']

    @staticmethod
    def collate_fn(data):
        origin_max_len = max(elem['origin_len'] for elem in data)
        masked_max_len = max(elem['masked_len'] for elem in data)

        origin = torch.zeros((len(data), origin_max_len), dtype=torch.long)
        masked = torch.zeros((len(data), masked_max_len), dtype=torch.long)

        for i, elem in enumerate(data):
            origin_elem = elem['origin']
            masked_elem = elem['masked']

            origin[i][:len(origin_elem)] = origin_elem
            masked[i][:len(masked_elem)] = masked_elem

        return {
            'origin': origin,
            'masked': masked
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.sentences)

        for p in self.batch(self.data(), self.batch_size * 10):
            yield from map(self.collate_fn, self.batch(sorted(p, key=lambda x: x['origin_len']), self.batch_size))

    def batch(self, data, batch_size):
        minibatch, size_so_far = [], 0

        for elem in data:
            minibatch.append(elem)
            size_so_far += self.batch_size_fn(elem)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(elem)
        if minibatch:
            yield minibatch
