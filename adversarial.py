import pandas as pd

from utils import create_mapping


class Adversarial1:

    def __init__(self):
        self.languages = ['en', 'de', 'es', 'fr', 'ja', 'pt', 'zh_cn']

        train = pd.DataFrame()

        for language in self.languages:
            df = pd.read_csv(
                f'data/atis/train/train_{language}.tsv',
                delimiter='\t',
                index_col='id'
            )
            df['language'] = language
            train = pd.concat((train, df))

        train.reset_index(drop=True, inplace=True)

        slot2idx, idx2slot, intent2idx = create_mapping(train)
        num_slots, num_intents = len(slot2idx), len(intent2idx)

    def calculate_loss(self):
        pass
