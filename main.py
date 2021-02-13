import json

import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from model import Model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 1234


def train(model, dataloader, optimizer):
    model.train()

    total_loss = 0

    for batch in tqdm(dataloader, desc='TRAINING'):
        batch = {key: batch[key].to(device) for key in batch.keys()}

        optimizer.zero_grad()

        output = model(**batch)

        total_loss += output.item()

        output.backward()

        optimizer.step()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0

    for batch in tqdm(dataloader, desc='EVALUATING'):
        batch = {key: batch[key].to(device) for key in batch.keys()}

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

    model = Model()
    model.load_models()

    model.to(device)
    model.set_nonrussian_grad_zero()
    model.russian_forward()

    with open('data/dstc_utterances.json') as f:
        data = json.load(f)

    texts, _ = zip(*((elem['text'], elem['intent']) for elem in data))
    texts_train, texts_test = train_test_split(texts, test_size=0.1, random_state=SEED)

    dataset_train = CustomDataset(texts_train)
    dataset_test = CustomDataset(texts_test)

    dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = Adam(model.parameters(), lr=1e-4)

    for i in range(3):
        loss_train = train(model, dataloader_train, optimizer)

        loss_test = evaluate(model, dataloader_test)

        print(loss_train, loss_test)

    model.save_models()


if __name__ == '__main__':
    main()
