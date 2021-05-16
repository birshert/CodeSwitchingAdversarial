import sys
import warnings

import torch
import wandb
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from dataset import prepare_mlm_datasets
from utils import _set_seed
from utils import load_config
from utils import model_mapping
from utils import set_global_logging_level


warnings.filterwarnings('ignore')

set_global_logging_level()

SEED = 1234


def main(config_path: str = 'config.yaml'):
    config = load_config(config_path)

    cuda_device = int('m-bert' in config['model_name'])

    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    _set_seed(SEED)

    print(
        'Using device {}'.format(
            torch.cuda.get_device_name() + f':{cuda_device}' if torch.cuda.is_available() else 'cpu'
        )
    )

    log = config['log']

    run_name = config['model_name']

    wandb.init(
        project='diploma',
        entity='birshert',
        mode='online' if log else 'disabled',
        save_code=True,
        name=run_name
    )
    wandb.config.update(config)

    model = model_mapping[wandb.config['model_name']](config=wandb.config)
    model.to(device, non_blocking=True)

    train_dataset, test_dataset, collator = prepare_mlm_datasets(wandb.config, model)

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=wandb.config['batch_size'],
        pin_memory=True, drop_last=False, collate_fn=collator
    )

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=wandb.config['batch_size'],
        pin_memory=True, drop_last=False, collate_fn=collator
    )

    optimizer = AdamW(model.parameters(), lr=wandb.config['learning_rate'])
    scaler = GradScaler(enabled=wandb.config['fp-16'])

    num_epoches = wandb.config['num_epoches']
    log_interval = wandb.config['log_interval']

    with tqdm(total=num_epoches * (len(train_loader) + len(test_loader) * wandb.config['log_metrics'] * log)) as p_bar:
        for epoch in range(num_epoches):
            if log and wandb.config['log_metrics']:
                model.eval()

                with torch.no_grad():
                    total_loss = 0

                    for i, batch in enumerate(test_loader):
                        p_bar.set_description(
                            f'EVALUATE EPOCH [{epoch + 1:02d}/{num_epoches:02d}], BATCH [{i + 1:03d}/'
                            f'{len(test_loader)}]'
                        )

                        with autocast(enabled=wandb.config['fp-16']):
                            batch = {key: batch[key].to(device, non_blocking=True) for key in batch.keys()}
                            loss, _ = model(**batch)

                            total_loss += loss.item()

                        p_bar.update()

                wandb.log(
                    {
                        'loss [VALID]': total_loss / len(test_loader)
                    }
                )

            model.train()

            for i, batch in enumerate(train_loader):
                p_bar.set_description(
                    f'TRAIN EPOCH [{epoch + 1:02d}/{num_epoches:02d}], BATCH [{i + 1:03d}/{len(train_loader)}]'
                )

                with autocast(enabled=wandb.config['fp-16']):
                    optimizer.zero_grad()

                    batch = {key: batch[key].to(device, non_blocking=True) for key in batch.keys()}
                    loss, _ = model(**batch)

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

                p_bar.update()

                if log and (i + epoch * len(train_loader)) % log_interval == 0:
                    wandb.log(
                        {
                            'loss [TRAIN]': loss.item(),
                            'step': i + epoch * len(train_loader)
                        }
                    )

            model.save()
            model.save_body()


if __name__ == '__main__':
    main(sys.argv[-1])
