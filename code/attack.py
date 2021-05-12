import os

import pandas as pd

from adversarial import AdversarialAlignments
from adversarial import AdversarialWordLevel
from adversarial import Pacifist
from utils import load_config


def main():
    config = load_config()

    languages = config['languages']

    pacifist = Pacifist()
    word_level_attacker = AdversarialWordLevel()
    alignments_attacker = AdversarialAlignments()

    word_level_attacker.num_examples = config['num_examples']
    alignments_attacker.num_examples = config['num_examples']

    try:
        base_language = 'en'
        print(f'Attacking language {base_language}')

        pacifist.change_base_language(base_language)
        word_level_attacker.change_base_language(base_language)
        alignments_attacker.change_base_language(base_language)

        base_path = f'results/{base_language}/'
        model_name = f'{config["model_name"]}_{int(config["only_english"])}_{int(config["load_adv_pretrained"])}.csv'

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        other_languages = list(languages)
        other_languages.remove(base_language)

        pacifist.port_model()

        results = {
            'No attack': pacifist.attack_dataset(),
        }

        pacifist.port_model('cpu')

        # WORD LEVEL

        word_level_attacker.port_model()

        for language in other_languages:
            word_level_attacker.change_attack_language(language)
            results[f'Word level [{language}]'] = word_level_attacker.attack_dataset()

        word_level_attacker.port_model('cpu')

        # ALIGNMENTS

        alignments_attacker.port_model()

        for language in other_languages:
            alignments_attacker.change_attack_language(language)
            results[f'Alignments [{language}]'] = alignments_attacker.attack_dataset()

        alignments_attacker.port_model('cpu')

        pd.DataFrame.from_dict(results).transpose().to_csv(base_path + model_name)
    except KeyboardInterrupt:
        try:
            pd.DataFrame.from_dict(results).transpose().to_csv(base_path + model_name)
        finally:
            exit(0)


if __name__ == '__main__':
    main()
