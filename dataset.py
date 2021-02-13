from abc import ABCMeta

import numpy as np
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from transformers import (
    XLMRobertaTokenizerFast,
)


class CustomDataset(Dataset, metaclass=ABCMeta):
    """
    Returns tokenized sentence and masked sentence (currently masking random word from the sentence), using XLM-R
    tokenizer.
    """

    def __init__(self, sentences):
        self.sentences = [
            word_tokenize(sentence) for sentence in sentences
        ]

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained('models/tokenizer')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item: int):
        sentence = ' '.join(self.sentences[item])

        word_to_mask = np.random.choice(np.arange(0, len(self.sentences[item])))

        masked_sentence = self.sentences[item]
        masked_sentence[word_to_mask] = self.tokenizer.mask_token
        masked_sentence = ' '.join(masked_sentence)

        return {
            'sentence': self.tokenizer.encode(sentence, return_tensors='pt', padding='max_length').flatten(),
            'masked_sentence': self.tokenizer.encode(masked_sentence, return_tensors='pt', padding='max_length').flatten()
        }
