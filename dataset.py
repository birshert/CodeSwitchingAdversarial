from abc import ABCMeta

import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class CustomDataset(Dataset, metaclass=ABCMeta):
    """
    Returns a sentence and a masked sentence (currently masking random noun from the sentence).
    """

    def __init__(self, sentences):
        self.sentences = [
            word_tokenize(sentence) for sentence in sentences
        ]

        self.noun_masks = [
            list(map(lambda x: x[1] == 'NN', pos_tag(sentence))) for sentence in self.sentences
        ]

        self.mask_token = '<mask>'

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item: int):
        sentence = ' '.join(self.sentences[item])

        word_to_mask = np.random.choice(np.nonzero(self.noun_masks[item])[0])

        masked_sentence = self.sentences[item]
        masked_sentence[word_to_mask] = self.mask_token
        masked_sentence = ' '.join(masked_sentence)

        return {
            'sentence': sentence,
            'masked_sentence': masked_sentence
        }
