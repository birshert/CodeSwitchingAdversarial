import sys
import warnings

import torch
import wandb
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from dataset import prepare_joint_datasets
from utils import _set_seed
from utils import compute_metrics
from utils import get_cuda_device
from utils import load_config
from utils import model_mapping
from utils import set_global_logging_level


warnings.filterwarnings('ignore')

set_global_logging_level()

SEED = 1234


@torch.no_grad()
def joint_evaluate(model, dataloader, device, p_bar=None, **kwargs):
    model.eval()

    total_loss = 0

    slot_preds = []
    intent_preds = torch.tensor([])

    slot_true = []
    intent_true = torch.tensor([])

    for i, batch in enumerate(dataloader):
        if p_bar is not None:
            p_bar.set_description(
                f'EVALUATE EPOCH [{kwargs["epoch"] + 1:02d}/{kwargs["num_epoches"]:02d}], BATCH [{i + 1:03d}/'
                f'{len(dataloader)}]'
            )

        with autocast(enabled=kwargs['fp_16']):
            batch = {key: batch[key].to(device, non_blocking=True) for key in batch.keys()}
            loss, intent_logits, slot_logits = model(**batch)

            total_loss += loss.item()

        slot_preds.extend(slot_logits.cpu())
        intent_preds = torch.cat((intent_preds, intent_logits.cpu()))

        slot_true.extend(batch['slot_labels_ids'].cpu())
        intent_true = torch.cat((intent_true, batch['intent_label_ids'].cpu()))

        if p_bar is not None:
            p_bar.update()

    slot_preds = pad_sequence(slot_preds, batch_first=True, padding_value=kwargs['slot2idx']['PAD'])
    slot_true = pad_sequence(slot_true, batch_first=True, padding_value=kwargs['slot2idx']['PAD'])

    slot_preds = slot_preds.argmax(dim=-1).numpy()
    intent_preds = intent_preds.argmax(dim=-1).numpy()

    slot_true = slot_true.numpy()
    intent_true = intent_true.numpy()

    slot_true_list = [[] for _ in range(slot_true.shape[0])]
    slot_preds_list = [[] for _ in range(slot_true.shape[0])]

    for i in range(slot_true.shape[0]):
        for j in range(slot_true.shape[1]):
            if slot_true[i, j]:
                slot_true_list[i].append(kwargs['idx2slot'][slot_true[i, j]])
                slot_preds_list[i].append(kwargs['idx2slot'][slot_preds[i, j]])

    results = {
        'loss': total_loss / len(dataloader),
    }

    results.update(compute_metrics(intent_preds, intent_true, slot_preds_list, slot_true_list))

    return results


def main(config_path: str = 'config.yaml'):
    config = load_config(config_path)

    device = get_cuda_device(config)
    _set_seed(SEED)

    print(
        'Using device {}'.format(
            torch.cuda.get_device_name() + f':{device}' if torch.cuda.is_available() else 'cpu'
        )
    )

    log = config['log']

    run_name = [config['model_name']]
    if config['only_english']:
        run_name += ['en']
    if config['load_adv_pretrained']:
        run_name += ['adv']

    run_name = ' '.join(run_name)

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

    train_dataset, test_dataset, collator, slot2idx, idx2slot = prepare_joint_datasets(wandb.config, model)

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
                evaluation_results = joint_evaluate(
                    model, test_loader, device,
                    p_bar, fp_16=wandb.config['fp-16'],
                    epoch=epoch, num_epoches=num_epoches,
                    slot2idx=slot2idx, idx2slot=idx2slot
                )

                evaluation_results['loss [VALID]'] = evaluation_results.pop('loss')

                wandb.log(evaluation_results)

            model.train()

            for i, batch in enumerate(train_loader):
                p_bar.set_description(
                    f'TRAIN EPOCH [{epoch + 1:02d}/{num_epoches:02d}], BATCH [{i + 1:03d}/{len(train_loader)}]'
                )

                with autocast(enabled=wandb.config['fp-16']):
                    optimizer.zero_grad()

                    batch = {key: batch[key].to(device, non_blocking=True) for key in batch.keys()}
                    loss, _, _ = model(**batch)

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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[-1])
    else:
        main()
