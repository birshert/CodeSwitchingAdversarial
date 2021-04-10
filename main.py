import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from dataset import prepare_datasets
from model import JointBERT
from utils import (
    MODEL_CLASSES,
    MODEL_PATH_MAP,
    set_global_logging_level,
)


set_global_logging_level()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 1234


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0
    dataloader_len = 0

    for batch in dataloader:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        b_input_ids, b_tags, b_intents, b_attention_masks = batch

        output = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_masks,
            intent_label_ids=b_intents,
            slot_labels_ids=b_tags
        )[0]

        total_loss += output.item()

        dataloader_len += 1

    model.train()

    return total_loss / dataloader_len


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
            'model_name': 'bert',
            'num_epoches': 5,
            'log_interval': 1000,
            'learning_rate': 1e-4,
            'batch_size': 8,
            'dropout': 0.1,
            'ignore_index': 0,
            'slot_coef': 1.0
        }
    )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[wandb.config['model_name']]
    model_path = MODEL_PATH_MAP[wandb.config['model_name']]

    train_dataset, test_dataset, num_slots, num_intents = prepare_datasets(
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
        pin_memory=True, num_workers=8, drop_last=False
    )

    test_loader = DataLoader(
        test_dataset, shuffle=True, batch_size=wandb.config['batch_size'],
        pin_memory=True, num_workers=8, drop_last=False
    )

    model = JointBERT.from_pretrained(
        model_path,
        config=config_class.from_pretrained(model_path),
        wandb_config=wandb.config
    )

    model.to(device, non_blocking=True)

    optimizer = AdamW(model.parameters(), lr=wandb.config['learning_rate'])

    num_epoches = wandb.config['num_epoches']
    log_interval = wandb.config['log_interval']

    with tqdm(total=num_epoches * len(train_loader)) as progress_bar:
        for epoch in range(num_epoches):
            for i, batch in enumerate(train_loader):
                progress_bar.set_description(
                    f'EPOCH [{epoch + 1:02d}/{num_epoches:02d}], BATCH [{i + 1:03d}/{len(train_loader)}]'
                )

                batch = tuple(t.to(device, non_blocking=True) for t in batch)
                b_input_ids, b_tags, b_intents, b_attention_masks = batch

                optimizer.zero_grad()

                loss = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_masks,
                    intent_label_ids=b_intents,
                    slot_labels_ids=b_tags
                )[0]

                loss.backward()

                optimizer.step()

                progress_bar.update()

                if log and (i + epoch * len(train_loader)) % log_interval == 0:
                    test_loss = evaluate(model, test_loader)
                    wandb.log(
                        {
                            'Loss [TRAIN]': loss.item(),
                            'Loss [TEST]': test_loss,
                        }
                    )


if __name__ == '__main__':
    main()
