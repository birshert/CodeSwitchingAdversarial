import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CustomLoss(nn.Module):
    """
    Custom loss counting similarity between target and output sentence
    """

    def __init__(self, device):
        super().__init__()

        self.model_name = 'sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens'
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()

    def forward(self, outputs, labels):
        outputs_1 = self.model(**outputs)
        outputs_2 = self.model(**labels)

        embeddings_1 = outputs_1.pooler_output
        embeddings_2 = outputs_2.pooler_output

        normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
        normalized_embeddings_2 = F.normalize(embeddings_2, p=2)

        return normalized_embeddings_1 @ normalized_embeddings_2.transpose(0, 1)
