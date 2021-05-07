import os

import pandas as pd

from adversarial import AdversarialAlignments
from adversarial import AdversarialWordLevel
from adversarial import Pacifist
from utils import load_config


def main():
    config = load_config()

    languages = config['languages']

    for base_language in languages:
        print(f'Attacking language {base_language}')

        base_path = f'results/{base_language}/'
        model_name = f'{config["model_name"]}_{int(config["load_pretrained"])}_{int(config["load_body"])}.csv'

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        other_languages = list(languages)
        other_languages.remove(base_language)

        results = {
            'No attack': Pacifist(base_language=base_language).attack_dataset(),
        }

        # WORD LEVEL

        for language in other_languages:
            results[f'Word level [{language}]'] = AdversarialWordLevel(
                base_language=base_language,
                languages=[language]
            ).attack_dataset()

        # ALIGNMENTS

        for language in other_languages:
            results[f'Alignments [{language}]'] = AdversarialAlignments(
                base_language=base_language,
                languages=[language]
            ).attack_dataset()

        pd.DataFrame.from_dict(results).transpose().to_csv(base_path + model_name)


if __name__ == '__main__':
    main()
