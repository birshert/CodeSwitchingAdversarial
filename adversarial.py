from copy import deepcopy

import numpy as np
import torch
import yaml
from word2word import Word2word

from utils import model_mapping


class BaseAdversarial:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        self.model = model_mapping[config['model_name']](config=config)
        self.model.to(self.device, non_blocking=True)

    def attack(self, x, y_slots, y_intents):
        if isinstance(x, str):
            x = x.split()

        if isinstance(y_slots, str):
            y_slots = y_slots.split()

        current_loss = self.calculate_loss(x, y_slots, y_intents)

        for pos in np.random.permutation(len(x)):
            candidates, losses = self.get_candidates(x, y_slots, y_intents, pos)

            if candidates and current_loss < np.max(losses):
                current_loss = np.max(losses)
                x[pos] = candidates[np.argmax(losses)]

        return x

    def get_candidates(self, x, y_slots, y_intents, pos):
        raise NotImplementedError

    def calculate_loss(self, x, y_slots, y_intents) -> float:
        return np.random.normal(0, 10)


class Adversarial1(BaseAdversarial):
    """
    Simple adversarial attack based on changing words in random order to their translations in set of target languages,
    that maximize model's loss.
    """

    def __init__(self, languages: list = None):
        super().__init__()

        if languages is None:
            languages = ['de', 'es', 'fr', 'ja', 'pt', 'zh_cn']

        self.translators = {lang: Word2word('en', lang) for lang in languages}

    def get_candidates(self, x, y_slots, y_intents, pos):
        xc = deepcopy(x)

        candidates, losses = [], []

        for lang in self.translators.keys():
            token = self.translators[lang](x[pos], n_best=1)[0]
            candidates.append(token)

            xc[pos] = token
            losses.append(self.calculate_loss(xc, y_slots, y_intents))

        return candidates, losses
