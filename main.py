import json

import torch
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AdamW

from dataset import CustomDataloader
from model import Model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 1234


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0

    for batch in dataloader:
        batch = {key: batch[key].to(device, non_blocking=True) for key in batch.keys()}

        output = model(**batch)

        total_loss += output.item()

    return total_loss / len(dataloader)


def _set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    _set_seed(SEED)
    print('Using device {}'.format(torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'))

    log = True

    wandb.init(project='diploma', entity='birshert', mode='online' if log else 'disabled', save_code=True)

    wandb.config.update(
        {
            'num_epoches': 1,
            'log_interval': 50,
            'log_examples': 500,
            'learning_rate': 1e-4,
            'batch_size': 1000
        }
    )

    model = Model()
    model.to(device, non_blocking=True)
    model.russian_forward()

    with open('data/dstc_utterances.json') as f:
        data = json.load(f)

    texts, _ = zip(*((elem['text'], elem['intent']) for elem in data))
    texts_train, texts_valid = train_test_split(texts, test_size=0.01, random_state=SEED)

    train_loader = CustomDataloader(texts_train, batch_size=wandb.config['batch_size'])
    valid_loader = CustomDataloader(texts_valid, batch_size=wandb.config['batch_size'], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=wandb.config['learning_rate'])

    num_epoches = wandb.config['num_epoches']
    log_interval = wandb.config['log_interval']
    log_interval_examples = wandb.config['log_examples']

    len_train_loader = 0

    for _ in train_loader:
        len_train_loader += 1

    with tqdm(total=num_epoches * len_train_loader) as progress_bar:
        for epoch in range(num_epoches):
            for i, batch in enumerate(train_loader):
                progress_bar.set_description(
                    f'EPOCH [{epoch + 1:02d}/{num_epoches:02d}], BATCH [{i + 1:03d}/{len_train_loader}]'
                )

                batch = {key: batch[key].to(device) for key in batch.keys()}

                optimizer.zero_grad()

                loss = model(**batch)

                loss.backward()

                optimizer.step()

                progress_bar.update()

                if log and (i + epoch * len(train_loader)) % log_interval == 0:
                    valid_loss = evaluate(model, valid_loader)
                    wandb.log(
                        {
                            'Semantic similarity loss [TRAIN]': loss.item(),
                            'Semantic similarity loss [VALID]': valid_loss.item(),
                        }
                    )

                # if log and (i + epoch * len(train_loader)) % log_interval_examples == 0:
                #     wandb.log(
                #         {
                #             f'Image step [{(i + epoch * len(train_loader))}]': [
                #                 wandb.Image(torch_2_image(glow.sample_fixed(64), rows=8))
                #             ]
                #         }
                #     )

        model.save()


if __name__ == '__main__':
    main()
