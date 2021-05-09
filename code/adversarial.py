from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from more_itertools import chunked
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomJointDataset
from dataset import JointCollator
from dataset import read_atis
from train_joint_model import joint_evaluate
from utils import create_mapping
from utils import load_config
from utils import model_mapping
from utils import tokenize_and_preserve_labels


CPU_COUNT = mp.cpu_count()


class BaseAdversarial:
    """
    Base class for adversarial attacks.

    Implements basic init + attack dataset + calculate loss functions.
    """

    def __init__(self, base_language: str = 'en', attack_language: str = None, init_model: bool = True):
        self.slot2idx, self.idx2slot, self.intent2idx = create_mapping(read_atis('train'))
        self.collator = JointCollator(self.slot2idx)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.config = load_config()

        if init_model:
            self.model = model_mapping[self.config['model_name']](config=self.config)
            self.model.load()
            self.model.eval()
            self.model.to(self.device, non_blocking=True)

        self.base_language = base_language
        self.num_examples = 1

        if attack_language is None:
            other_languages = self.config['languages']
            other_languages.remove(self.base_language)
            attack_language = np.random.choice(other_languages)

        self.attack_language = attack_language

        self.rng = np.random.default_rng()

    def get_tokens(self, x, pos, *args) -> list[str]:
        raise NotImplementedError

    def get_candidates(self, x, y_slots, y_intent, pos, *args) -> list:
        """
        Performs attack on a single example.
        :return: adversarial perturbation.
        """
        xc = deepcopy(x)
        y_slots_c = deepcopy(y_slots)

        tokens = self.get_tokens(x, pos, args)

        if tokens is None:
            return x, y_slots, y_intent

        if len(tokens) > 1:
            if y_slots[pos].startswith('B'):
                new_slot_label = 'I' + y_slots[pos][1:]
                y_slots_c[pos] = y_slots[pos] + ' '.join(new_slot_label for _ in range(len(tokens) - 1))
            else:
                y_slots_c[pos] = ' '.join(y_slots[pos] for _ in range(len(tokens)))

        xc[pos] = ' '.join(tokens)

        return ' '.join(xc).split(), ' '.join(y_slots_c).split(), y_intent

    def attack(self, x, y_slots, y_intent, *args) -> tuple[list[list[str]], list[list[str]], list[str], list[float]]:
        num_objects = len(x)

        for idx in range(num_objects):
            x[idx] = x[idx].split()
            y_slots[idx] = y_slots[idx].split()

        current_loss = self.calculate_loss(x, y_slots, y_intent)

        for pos in np.array(
                [
                    self.rng.permutation(len(x[0])) for _ in range(num_objects)
                ]
        ).T:  # choosing indexes in random order

            candidates = list(map(self.get_candidates, *(x, y_slots, y_intent, pos, *args)))

            losses = self.calculate_loss(*list(zip(*candidates)))

            for idx in range(num_objects):
                if candidates[idx] and losses[idx] > current_loss[idx]:  # if we can "improve" loss
                    current_loss[idx] = losses[idx]
                    x[idx], y_slots[idx], y_intent[idx] = candidates[idx]

        return x, y_slots, y_intent, current_loss

    @torch.no_grad()
    def calculate_loss(self, x, y_slots, y_intent) -> list[float]:
        """
        Calculates loss of model on an example.
        :param x: example utterance.
        :param y_slots: example slot labels.
        :param y_intent: example intent label.
        :return: float value of loss.
        """

        data = []

        for idx in range(len(x)):
            tokens, slot_labels = tokenize_and_preserve_labels(
                self.model.tokenizer, x[idx], y_slots[idx], self.slot2idx
            )

            input_ids = torch.tensor(
                self.model.tokenizer.convert_tokens_to_ids(tokens),
                dtype=torch.long,
            )

            slot_labels = torch.tensor(
                slot_labels,
                dtype=torch.long,
            )

            intent = torch.tensor(
                self.intent2idx.get(y_intent[idx], self.intent2idx['UNK']),
                dtype=torch.long,
            )

            data.append((input_ids, slot_labels, intent))

        batch = self.collator(data)
        batch = {key: batch[key].to(self.device) for key in batch.keys()}

        losses = self.model.calculate_loss(**batch)

        return losses.cpu().tolist()

    @torch.no_grad()
    def attack_dataset(self, subset: str = 'test') -> dict[str, float]:
        """
        Attacks atis subset.
        :param subset: atis subset.
        :return: evaluation results.
        """
        dataset = read_atis(subset, [self.base_language])
        dataset['len'] = dataset['utterance'].str.split().apply(len)

        data = []

        starting_time = time()

        perplexity = 0

        with tqdm(desc='GENERATING ADVERSARIAL EXAMPLES', total=len(dataset)) as progress_bar:
            for _ in range(self.num_examples):
                for key, group in dataset.groupby('len'):
                    idxes = list(chunked(group.index.values.tolist(), 16))

                    for i in idxes:
                        chunk = group.loc[i]

                        x = chunk['utterance'].values
                        y_slots = chunk['slot_labels'].values
                        y_intent = chunk['intent'].values

                        x, y_slots, y_intent, losses = self.attack(x, y_slots, y_intent)

                        perplexity += np.sum(np.exp(losses))

                        for idx in range(len(x)):
                            tokens, slot_labels = tokenize_and_preserve_labels(
                                self.model.tokenizer,
                                x[idx],
                                y_slots[idx],
                                self.slot2idx
                            )

                            data.append(
                                (
                                    tokens,
                                    slot_labels,
                                    self.intent2idx.get(y_intent[idx], self.intent2idx['UNK'])
                                )
                            )

                        progress_bar.update(len(i))

        data = CustomJointDataset(data, self.model.tokenizer, self.slot2idx)
        loader = DataLoader(data, batch_size=8, drop_last=False, collate_fn=JointCollator(self.slot2idx))

        results = joint_evaluate(
            self.model, loader, fp_16=True,
            slot2idx=self.slot2idx, idx2slot=self.idx2slot
        )

        results['perplexity'] = perplexity / (len(dataset) * self.num_examples)
        results['time'] = time() - starting_time

        return results


class Pacifist(BaseAdversarial):
    """
    No adversarial attack (passing examples through).
    """

    def __init__(self, base_language: str = 'en', attack_language: str = None, init_model: bool = True):
        super().__init__(base_language, attack_language, init_model)

        self.num_examples = 1

    def get_tokens(self, x, pos, *args) -> list[str]:
        pass

    def get_candidates(self, x, y_slots, y_intent, pos, **kwargs) -> tuple[list, list]:
        pass

    def attack(self, x, y_slots, y_intent, *args) -> tuple[list[list[str]], list[list[str]], list[str], list[float]]:
        num_objects = len(x)

        for idx in range(num_objects):
            x[idx] = x[idx].split()
            y_slots[idx] = y_slots[idx].split()

        return x, y_slots, y_intent, self.calculate_loss(x, y_slots, y_intent)


class AdversarialWordLevel(BaseAdversarial):
    """
    Simple adversarial attack based on changing tokens to their translations in SET of target languages.
    Token's translation is chosen in order to maximize model's loss.
    Translations are generated with dictionaries from word2word library.
    """

    def __init__(self, base_language: str = 'en', attack_language: str = None, init_model: bool = True):
        super().__init__(base_language, attack_language, init_model)

        self.translations = torch.load('data/atis_test_translations/translations.pt')

    def get_tokens(self, x, pos, *args) -> list[str]:
        try:
            return self.translations[self.base_language][self.attack_language][x[pos]].split()
        except KeyError:
            return None


def mapping_alignments(lines, data) -> dict[int, dict[int, str]]:
    """
    Create mapping for alignments and dataset (alignments should be for this dataset).
    :param lines: lines generated with awesome-align.
    :param data: array with texts.
    :return: dict with alignments: (line_idx: dict[token_idx: list[str(tokens)]]).
    """
    if len(lines) != len(data):
        raise ValueError('Alignments should be from this data.')

    mapping: dict = {}

    for idx, line in enumerate(lines):
        mapping[idx] = defaultdict(list)
        text = data[idx].strip().split()

        for elem in line.split():
            key, value = map(int, elem.split('-'))
            mapping[idx][key].append(text[value])

        mapping[idx] = dict(mapping[idx])

    return mapping


class AdversarialAlignments(BaseAdversarial):
    """
    More complicated adversarial attack based on changing tokens to their "alignment substitutions" in a
    SET of languages. Alignments are precomputed. Candidates are chosen in order to maximize model's loss.
    """

    def __init__(
            self, base_language: str = 'en', attack_language: str = None,
            init_model: bool = True, subset: str = 'test'
    ):
        super().__init__(base_language, attack_language, init_model)

        self.alignments: dict[int, dict[int, str]] = {}

        with open(f'data/atis_{subset}_alignment/{self.base_language}_{self.attack_language}.out') as f:
            self.alignments = mapping_alignments(
                f.readlines(),
                read_atis(subset, [self.attack_language])['utterance']
            )

    def get_tokens(self, x, pos, *args) -> list[str]:
        alignments = args[0][0]

        try:
            return alignments[pos]
        except KeyError:
            return None

    @torch.no_grad()
    def attack_dataset(self, subset: str = 'test') -> dict[str, float]:
        """
        Attacks atis subset.
        :param subset: atis subset.
        :return: evaluation results.
        """
        dataset = read_atis(subset, [self.base_language])
        dataset['len'] = dataset['utterance'].str.split().apply(len)

        data = []

        starting_time = time()

        perplexity = 0

        with tqdm(desc='GENERATING ADVERSARIAL EXAMPLES', total=len(dataset)) as progress_bar:
            for _ in range(self.num_examples):
                for key, group in dataset.groupby('len'):
                    idxes = list(chunked(group.index.values.tolist(), 16))

                    for i in idxes:
                        chunk = group.loc[i]

                        x = chunk['utterance'].values
                        y_slots = chunk['slot_labels'].values
                        y_intent = chunk['intent'].values
                        alignments = [self.alignments[ii] for ii in i]

                        x, y_slots, y_intent, losses = self.attack(x, y_slots, y_intent, alignments)

                        perplexity += np.sum(np.exp(losses))

                        for idx in range(len(x)):
                            tokens, slot_labels = tokenize_and_preserve_labels(
                                self.model.tokenizer,
                                x[idx],
                                y_slots[idx],
                                self.slot2idx
                            )

                            data.append(
                                (
                                    tokens,
                                    slot_labels,
                                    self.intent2idx.get(y_intent[idx], self.intent2idx['UNK'])
                                )
                            )

                        progress_bar.update(len(i))

        data = CustomJointDataset(data, self.model.tokenizer, self.slot2idx)
        loader = DataLoader(data, batch_size=8, drop_last=False, collate_fn=JointCollator(self.slot2idx))

        results = joint_evaluate(
            self.model, loader, fp_16=True,
            slot2idx=self.slot2idx, idx2slot=self.idx2slot
        )

        results['perplexity'] = perplexity / (len(dataset) * self.num_examples)
        results['time'] = time() - starting_time

        return results


class RandomAdversarialAlignments(AdversarialAlignments):
    """
    Adversarial attack, performing random changes in data (based on AdversarialAlignments attack).
    """

    def __init__(
            self, base_language: str = 'en', attack_language: str = None, init_model: bool = False,
            subset: str = 'test', perturbation_probability: float = 0.5, num_examples: int = 1
    ):
        super().__init__(base_language, attack_language, subset=subset, init_model=init_model)

        self.num_examples = num_examples
        self.perturbation_probability = perturbation_probability

    @torch.no_grad()
    def generate_dataset(self, subset: str = 'test'):
        dataset = read_atis(subset, [self.base_language])

        data = []

        for idx, row in tqdm(dataset.iterrows(), desc='GENERATING ADVERSARIAL EXAMPLES', total=len(dataset)):
            x = row['utterance']
            y_slots = row['slot_labels']
            y_intent = row['intent']
            alignments = self.alignments[idx]

            for _ in range(self.num_examples):
                example = self.attack(x, y_slots, y_intent, alignments)

                data.append(
                    {
                        'utterance': ' '.join(example),
                        'slot_labels': y_slots,
                        'intent': y_intent
                    }
                )

        return pd.DataFrame.from_dict(data)

    def attack(self, x, y_slots, y_intent, *args) -> list[str]:
        if isinstance(x, str):
            x = x.split()

        if isinstance(y_slots, str):
            y_slots = y_slots.split()

        for pos in self.rng.permutation(len(x)):
            candidates = self.get_candidates(x, y_slots, y_intent, pos, args)

            if candidates and self.rng.uniform() > self.perturbation_probability:
                x, y_slots, y_intent = candidates

        return x

    def get_candidates(self, x, y_slots, y_intent, pos, *args) -> list:
        xc = deepcopy(x)
        y_slots_c = deepcopy(y_slots)

        tokens = self.get_tokens(x, pos, args)

        if tokens is not None:
            if len(tokens) > 1:
                if y_slots[pos].startswith('B'):
                    new_slot_label = 'I' + y_slots[pos][1:]
                    y_slots_c[pos] = y_slots[pos] + ' '.join(new_slot_label for _ in range(len(tokens) - 1))
                else:
                    y_slots_c[pos] = ' '.join(y_slots[pos] for _ in range(len(tokens)))

            xc[pos] = ' '.join(tokens)

        return xc, y_slots_c, y_intent
