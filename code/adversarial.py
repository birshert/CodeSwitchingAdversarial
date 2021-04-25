from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from word2word import Word2word

from dataset import (
    CustomDataset,
    read_atis,
)
from train_model import evaluate
from utils import (
    create_mapping,
    load_config,
    model_mapping,
    tokenize_and_preserve_labels,
)


@torch.no_grad()
def evaluate_adversarial(attacker_class, **kwargs):
    attacker = attacker_class(**kwargs)

    test = read_atis('test', ['en'])

    data = []

    for idx, row in tqdm(test.iterrows(), desc='GENERATING ADVERSARIAL EXAMPLES', total=len(test)):

        x = row['utterance']
        y_slots = row['slot_labels']
        y_intent = row['intent']

        tokens, slot_labels = tokenize_and_preserve_labels(
            attacker.model.tokenizer,
            ' '.join(attacker.attack(x, y_slots, y_intent)),
            y_slots,
            attacker.slot2idx
        )

        data.append(
            (
                tokens,
                slot_labels,
                attacker.intent2idx.get(y_intent, attacker.intent2idx['UNK'])
            )
        )

    dataset = CustomDataset(data, attacker.model.tokenizer, attacker.slot2idx)
    loader = DataLoader(
        dataset, shuffle=True, batch_size=8,
        pin_memory=True, drop_last=False, collate_fn=dataset.collate_fn
    )

    return evaluate(
        attacker.model, loader, fp_16=True,
        epoch=0, num_epoches=1,
        slot2idx=attacker.slot2idx, idx2slot=attacker.idx2slot
    )


class BaseAdversarial:

    def __init__(self):
        self.slot2idx, self.idx2slot, self.intent2idx = create_mapping(read_atis('train'))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.config = load_config()

        self.model = model_mapping[self.config['model_name']](config=self.config)
        self.model.to(self.device, non_blocking=True)

    def get_candidates(self, x, y_slots, y_intent, pos):
        raise NotImplementedError

    @torch.no_grad()
    def calculate_loss(self, x, y_slots, y_intent) -> float:
        tokens, slot_labels = tokenize_and_preserve_labels(
            self.model.tokenizer, x, y_slots, self.slot2idx
        )

        input_ids = torch.tensor(
            self.model.tokenizer.convert_tokens_to_ids(tokens),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        intent = torch.tensor(
            self.intent2idx.get(y_intent, self.intent2idx['UNK']),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        slot_labels = torch.tensor(
            slot_labels,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
            device=self.device
        )

        loss = self.model(input_ids, intent, slot_labels, attention_mask)

        return loss[0].cpu().item()


class Adversarial1(BaseAdversarial):
    """
    Simple adversarial attack based on changing tokens to their translations in set of target languages.
    Token's translation is chosen in order to maximize model's loss.
    Translations are generated with dictionaries from word2word library.
    """

    def __init__(self, languages: list = None):
        super().__init__()

        if languages is None:
            languages = self.config['languages']
            languages.remove('en')

        self.languages = languages

        self.translators = {lang: Word2word('en', lang) for lang in languages}

    def attack(self, x, y_slots, y_intent):
        if isinstance(x, str):
            x = x.split()

        if isinstance(y_slots, str):
            y_slots = y_slots.split()

        current_loss = self.calculate_loss(x, y_slots, y_intent)

        for pos in np.random.permutation(len(x)):
            candidates, losses = self.get_candidates(x, y_slots, y_intent, pos)

            if candidates and current_loss < np.max(losses):
                current_loss = np.max(losses)
                x[pos] = candidates[np.argmax(losses)]

        return x

    def get_candidates(self, x, y_slots, y_intent, pos):
        xc = deepcopy(x)

        candidates, losses = [], []

        for lang in self.languages:
            try:
                token = np.random.choice(self.translators[lang](x[pos], n_best=3))
                xc[pos] = token

                losses.append(self.calculate_loss(xc, y_slots, y_intent))
                candidates.append(token)
            except KeyError:
                pass

        return candidates, losses


class Adversarial2(Adversarial1):
    """
    Simple adversarial attack based on changing words in random order to their translations in set of target languages,
    that maximize model's loss.
    """

    def __init__(self, languages: list = None):
        super().__init__(languages)

        self.attack_language = None

    def attack(self, x, y_slots, y_intent, language=None):
        if language is None:
            self.attack_language = np.random.choice(self.languages)
        elif language in self.languages:
            self.attack_language = language
        else:
            raise ValueError(f'Language {language} is not listed as one of the languages used.')

        return super().attack(x, y_slots, y_intent)

    def get_candidates(self, x, y_slots, y_intent, pos):
        xc = deepcopy(x)

        candidates, losses = [], []

        try:
            token = np.random.choice(self.translators[self.attack_language](x[pos], n_best=3))
            xc[pos] = token

            losses.append(self.calculate_loss(xc, y_slots, y_intent))
            candidates.append(token)
        except KeyError:
            pass

        return candidates, losses
