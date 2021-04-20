import warnings

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from dataset import prepare_datasets
from model import XLMRoberta
from utils import (
    compute_metrics,
    MODEL_CLASSES,
    MODEL_PATH_MAP,
    set_global_logging_level,
)


warnings.filterwarnings('ignore')

set_global_logging_level()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 1234


@torch.no_grad()
def evaluate(model, dataloader, idx2slot, p_bar, **kwargs):
    model.eval()

    total_loss = 0

    slot_preds = torch.tensor([])
    intent_preds = torch.tensor([])

    slot_true = torch.tensor([])
    intent_true = torch.tensor([])

    for i, batch in enumerate(dataloader):
        p_bar.set_description(
            f'EVALUATE EPOCH [{kwargs["epoch"] + 1:02d}/{kwargs["num_epoches"]:02d}], BATCH [{i + 1:03d}/'
            f'{len(dataloader)}]'
        )

        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        batch = {
            'input_ids': batch[0],
            'slot_labels_ids': batch[1],
            'intent_label_ids': batch[2],
            'attention_mask': batch[3],
        }

        loss, intent_logits, slot_logits = model(**batch)

        total_loss += loss.item()

        slot_preds = torch.cat((slot_preds, slot_logits.cpu()))
        intent_preds = torch.cat((intent_preds, intent_logits.cpu()))

        slot_true = torch.cat((slot_true, batch['slot_labels_ids'].cpu()))
        intent_true = torch.cat((intent_true, batch['intent_label_ids'].cpu()))

        p_bar.update()

    slot_preds = slot_preds.argmax(dim=-1).numpy()
    intent_preds = intent_preds.argmax(dim=-1).numpy()

    slot_true = slot_true.numpy()
    intent_true = intent_true.numpy()

    slot_true_list = [[] for _ in range(slot_true.shape[0])]
    slot_preds_list = [[] for _ in range(slot_true.shape[0])]

    for i in range(slot_true.shape[0]):
        for j in range(slot_true.shape[1]):
            if slot_true[i, j]:
                slot_true_list[i].append(idx2slot[slot_true[i, j]])
                slot_preds_list[i].append(idx2slot[slot_preds[i, j]])

    results = {
        'loss [VALID]': total_loss / len(dataloader)
    }

    results.update(compute_metrics(intent_preds, intent_true, slot_preds_list, slot_true_list))

    return results


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
            'model_name': 'xlm-r',
            'num_epoches': 10,
            'log_interval': 50,
            'log_metrics': True,
            'learning_rate': 1e-5,
            'batch_size': 8,
            'dropout': 0,
            'ignore_index': 0,
            'slot_coef': 1.0
        }
    )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[wandb.config['model_name']]
    model_path = MODEL_PATH_MAP[wandb.config['model_name']]

    train_dataset, test_dataset, num_slots, num_intents, idx2slot = prepare_datasets(
        tokenizer_class.from_pretrained(model_path)
    )

    wandb.config.update(
        {
            'num_intent_labels': num_intents,
            'num_slot_labels': num_slots
        }
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=wandb.config['batch_size'],
        pin_memory=True, drop_last=False
    )

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=wandb.config['batch_size'],
        pin_memory=True, drop_last=False
    )

    model = XLMRoberta(
        model_path,
        config=config_class.from_pretrained(model_path),
        wandb_config=wandb.config
    )

    model.to(device, non_blocking=True)

    optimizer = AdamW(model.parameters(), lr=wandb.config['learning_rate'])

    num_epoches = wandb.config['num_epoches']
    log_interval = wandb.config['log_interval']

    with tqdm(total=num_epoches * (len(train_loader) + len(test_loader) * wandb.config['log_metrics'] * log)) as p_bar:
        for epoch in range(num_epoches):
            if log and wandb.config['log_metrics']:
                wandb.log(evaluate(model, test_loader, idx2slot, p_bar, epoch=epoch, num_epoches=num_epoches))

            model.train()

            for i, batch in enumerate(train_loader):
                p_bar.set_description(
                    f'TRAIN EPOCH [{epoch + 1:02d}/{num_epoches:02d}], BATCH [{i + 1:03d}/{len(train_loader)}]'
                )

                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                batch = {
                    'input_ids': batch[0],
                    'slot_labels_ids': batch[1],
                    'intent_label_ids': batch[2],
                    'attention_mask': batch[3],
                }

                optimizer.zero_grad()

                loss, _, _ = model(**batch)

                loss.backward()

                optimizer.step()

                p_bar.update()

                if log and (i + epoch * len(train_loader)) % log_interval == 0:
                    wandb.log(
                        {
                            'loss [TRAIN]': loss.item()
                        }
                    )


if __name__ == '__main__':
    main()
