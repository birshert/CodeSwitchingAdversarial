from copy import deepcopy

import numpy as np
import torch
from word2word import Word2word

from dataset import prepare_datasets
from model import XLMRoberta
from utils import (
    MODEL_CLASSES,
    MODEL_PATH_MAP,
)


class BaseAdversarial:

    def __init__(self):
        pass
        # wandb_config = {
        #     'model_name': 'xlm-r',
        #     'load_pretrained': False,
        #     'load_checkpoint': False,
        #     'num_epoches': 10,
        #     'log_interval': 50,
        #     'log_metrics': True,
        #     'learning_rate': 1e-5,
        #     'batch_size': 8,
        #     'dropout': 0.1,
        #     'ignore_index': 0,
        #     'slot_coef': 1.0,
        #     'fp-16': True
        # }
        #
        # config_class, model_class, tokenizer_class = MODEL_CLASSES[wandb_config['model_name']]
        # model_path = MODEL_PATH_MAP[wandb_config['model_name']]
        #
        # self.tokenizer = tokenizer_class.from_pretrained(model_path)
        #
        # _, _, num_slots, num_intents, slot2idx, idx2slot = prepare_datasets(self.tokenizer)
        #
        # wandb_config.update(
        #     {
        #         'num_intent_labels': num_intents,
        #         'num_slot_labels': num_slots
        #     }
        # )
        #
        # self.model = XLMRoberta(
        #     config=config_class.from_pretrained(model_path),
        #     wandb_config=wandb_config,
        #     model_path=model_path
        # )
        # self.model.load_state_dict(torch.load(f'models/{wandb_config["model_name"]}.pt'))

    def attack(self, x, y_slots, y_intents):
        if isinstance(x, str):
            x = x.split()

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
