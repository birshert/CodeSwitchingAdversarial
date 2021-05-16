from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
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


class BaseAdversarial:
    """
    Base class for adversarial attacks.

    Implements basic init + attack dataset + calculate loss functions.
    """

    def __init__(
            self, base_language: str = 'en', attack_language: str = None,
            init_model: bool = True, config_path: str = 'config.yaml'
    ):
        self.slot2idx, self.idx2slot, self.intent2idx = create_mapping(read_atis('train', ['en']))
        self.idx2intent = {value: key for key, value in self.intent2idx.items()}
        self.collator = JointCollator(self.slot2idx)

        self.config = load_config(config_path)

        cuda_device = min(int('m-bert' in self.config['model_name']), torch.cuda.device_count() - 1)

        self.device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

        if init_model:
            self.model = model_mapping[self.config['model_name']](config=self.config)
            self.model.load()
            self.model.eval()

        self.base_language = base_language

        if attack_language is None:
            other_languages = self.config['languages']
            other_languages.remove(self.base_language)
            attack_language = np.random.choice(other_languages)

        self.attack_language = attack_language

        self.num_examples = 1

        self.rng = np.random.default_rng()

    def port_model(self, device: str = 'cuda'):
        if device == 'cuda':
            self.model.to(self.device)
        else:
            self.model.cpu()

    def change_attack_language(self, new_language: str):
        self.attack_language = new_language

    def change_base_language(self, new_language: str):
        self.base_language = new_language

    def get_tokens(self, x, pos, *args):
        raise NotImplementedError

    def get_candidates(self, x, y_slots, y_intent, pos, *args):
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

        return xc, y_slots_c, y_intent

    def attack(self, x, y_slots, y_intent, *args):
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
    def calculate_loss(self, x, y_slots, y_intent):
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
    def predict(self, x, y_slots, y_intent):
        tokens, slot_labels = tokenize_and_preserve_labels(
            self.model.tokenizer, x, y_slots, self.slot2idx
        )

        data = []

        input_ids = torch.tensor(
            self.model.tokenizer.convert_tokens_to_ids(tokens),
            dtype=torch.long,
        )

        slot_labels = torch.tensor(
            slot_labels,
            dtype=torch.long,
        )

        intent = torch.tensor(
            self.intent2idx.get(y_intent, self.intent2idx['UNK']),
            dtype=torch.long,
        )

        data.append((input_ids, slot_labels, intent))

        batch = self.collator(data)
        batch = {key: batch[key].to(self.device) for key in batch.keys()}

        loss, intent_logits, slot_logits = self.model(**batch)

        intent_true = intent.item()
        slot_true = slot_labels

        intent_pred = intent_logits.cpu().argmax(dim=-1)[0].item()
        slot_preds = slot_logits.cpu().argmax(dim=-1)[0]

        return (
                   self.idx2intent[intent_true],
                   self.idx2intent[intent_pred],
                   list(map(lambda s: self.idx2slot[s.item()], slot_true)),
                   list(map(lambda s: self.idx2slot[s.item()], slot_preds))
               )

    @torch.no_grad()
    def attack_dataset(self, subset: str = 'test'):
        """
        Attacks atis subset.
        :param subset: atis subset.
        :return: evaluation results.
        """
        dataset = read_atis(subset, [self.base_language])
        dataset['len'] = dataset['utterance'].str.split().apply(len)

        data = []

        with tqdm(desc='GENERATING ADVERSARIAL EXAMPLES', total=len(dataset) * self.num_examples) as progress_bar:
            for _ in range(self.num_examples):
                for key, group in dataset.groupby('len'):
                    idxes = list(chunked(group.index.values.tolist(), 16))

                    for i in idxes:
                        chunk = group.loc[i]

                        x = chunk['utterance'].values
                        y_slots = chunk['slot_labels'].values
                        y_intent = chunk['intent'].values

                        x, y_slots, y_intent, losses = self.attack(x, y_slots, y_intent)

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
            self.model, loader, self.device, fp_16=True,
            slot2idx=self.slot2idx, idx2slot=self.idx2slot
        )

        results = {
            'intent_acc': results['intent_acc'],
            'slot_f1': results['slot_f1'],
            'sementic_frame_acc': results['sementic_frame_acc'],
            'loss': results['loss']
        }

        return results


class Pacifist(BaseAdversarial):
    """
    No adversarial attack (passing examples through).
    """

    def __init__(
            self, base_language: str = 'en', attack_language: str = None,
            init_model: bool = True, config_path: str = 'config.yaml'
    ):
        super().__init__(base_language, attack_language, init_model, config_path)

        self.num_examples = 1

    def get_tokens(self, x, pos, *args):
        pass

    def get_candidates(self, x, y_slots, y_intent, pos, **kwargs):
        pass

    def attack(self, x, y_slots, y_intent, *args):
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

    def __init__(
            self, base_language: str = 'en', attack_language: str = None,
            init_model: bool = True, config_path: str = 'config.yaml'
    ):
        super().__init__(base_language, attack_language, init_model, config_path)

        self.translations = torch.load('data/atis_test_translations/translations.pt')

    def get_tokens(self, x, pos, *args):
        try:
            return self.translations[self.base_language][self.attack_language][x[pos]].split()
        except KeyError:
            return None


def mapping_alignments(lines, data):
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
            init_model: bool = True, config_path: str = 'config.yaml', subset: str = 'test'
    ):
        super().__init__(base_language, attack_language, init_model, config_path)

        self.subset = subset

        self.alignments: dict[int, dict[int, str]] = {}

    def read_alignments(self):
        with open(f'data/atis_{self.subset}_alignment/{self.base_language}_{self.attack_language}.out') as f:
            self.alignments = mapping_alignments(
                f.readlines(),
                read_atis(self.subset, [self.attack_language])['utterance']
            )

    def change_attack_language(self, new_language: str):
        super().change_attack_language(new_language)
        self.read_alignments()

    def get_tokens(self, x, pos, *args):
        alignments = args[0][0]

        try:
            return alignments[pos]
        except KeyError:
            return None

    @torch.no_grad()
    def attack_dataset(self, subset: str = 'test'):
        """
        Attacks atis subset.
        :param subset: atis subset.
        :return: evaluation results.
        """
        subset = self.subset

        dataset = read_atis(subset, [self.base_language])
        dataset['len'] = dataset['utterance'].str.split().apply(len)

        data = []

        with tqdm(desc='GENERATING ADVERSARIAL EXAMPLES', total=len(dataset) * self.num_examples) as progress_bar:
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
            self.model, loader, self.device, fp_16=True,
            slot2idx=self.slot2idx, idx2slot=self.idx2slot
        )

        results = {
            'intent_acc': results['intent_acc'],
            'slot_f1': results['slot_f1'],
            'sementic_frame_acc': results['sementic_frame_acc'],
            'loss': results['loss']
        }

        return results


class RandomAdversarialAlignments(AdversarialAlignments):
    """
    Adversarial attack, performing random changes in data (based on AdversarialAlignments attack).
    """

    def __init__(
            self, base_language: str = 'en', attack_language: str = None,
            init_model: bool = True, config_path: str = 'config.yaml', subset: str = 'test',
            perturbation_probability: float = 0.5, num_examples: int = 1
    ):
        super().__init__(base_language, attack_language, init_model, config_path, subset)

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

    def attack(self, x, y_slots, y_intent, *args):
        if isinstance(x, str):
            x = x.split()

        if isinstance(y_slots, str):
            y_slots = y_slots.split()

        for pos in self.rng.permutation(len(x)):
            candidates = self.get_candidates(x, y_slots, y_intent, pos, args)

            if candidates and self.rng.uniform() > self.perturbation_probability:
                x, y_slots, y_intent = candidates

        return x

    def get_candidates(self, x, y_slots, y_intent, pos, *args):
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
