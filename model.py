import numpy as np
import regex
import torch
import torch.nn as nn
import torch.nn.functional as func
from transformers import (
    PretrainedConfig,
    XLMRobertaForMaskedLM,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)


def is_cyrillic(s: str):
    return bool(regex.search(r'\p{IsCyrillic}', s))


class Model(nn.Module):
    """
    MLM generating a token for origin and then another model calculating two sentences embeddings (origin and
    transformed).
    Calculating loss based on distance between embeddings.
    """

    def __init__(self):
        super().__init__()
        self.mlm_model_name = 'xlm-roberta-base'
        self.emb_model_name = 'xlm-roberta-base-mean-tokens'

        # self.mlm_model = XLMRobertaForMaskedLM(PretrainedConfig.from_json_file(f'models/{self.mlm_model_name}.json'))
        # self.emb_model = XLMRobertaModel(PretrainedConfig.from_json_file(f'models/{self.emb_model_name}.json'))

        self.mlm_model = XLMRobertaForMaskedLM.from_pretrained(self.mlm_model_name)
        self.emb_model = XLMRobertaModel.from_pretrained(self.emb_model_name)

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.mlm_model_name)
        self.vocab_len = len(self.tokenizer.get_vocab())

        self.emb_model.eval()

        for parameter in self.emb_model.parameters():
            parameter.requires_grad = False

        self.russian_tokens_mask = np.zeros(self.mlm_model.lm_head.decoder.weight.shape[0])

        for token in range(len(self.russian_tokens_mask)):
            if is_cyrillic(self.tokenizer.decode([token])):
                self.russian_tokens_mask[token] = 1

        self.russian_tokens_mask = self.russian_tokens_mask.astype(bool)

    def load(self):
        self.mlm_model.load_state_dict(torch.load(f'models/{self.mlm_model_name}.pth'))
        self.emb_model.load_state_dict(torch.load(f'models/{self.emb_model_name}.pth'))

    def save(self):
        torch.save(self.mlm_model.state_dict(), f'models/{self.mlm_model_name}_.pth')
        torch.save(self.emb_model.state_dict(), f'models/{self.emb_model_name}_.pth')

    @torch.no_grad()
    def russian_forward(self):
        self.mlm_model.lm_head.decoder.weight[~self.russian_tokens_mask] = 0
        self.mlm_model.lm_head.decoder.bias[~self.russian_tokens_mask] = 0

    def russian_hook(self):
        weight_multi = torch.zeros_like(self.mlm_model.lm_head.decoder.weight, device=self.mlm_model.device)
        weight_multi[self.russian_tokens_mask] = 1.0
        self.mlm_model.lm_head.decoder.weight.register_hook(lambda grad: grad.mul_(weight_multi))

        bias_multi = torch.zeros_like(self.mlm_model.lm_head.decoder.bias, device=self.mlm_model.device)
        bias_multi[self.russian_tokens_mask] = 1.0
        self.mlm_model.lm_head.decoder.bias.register_hook(lambda grad: grad.mul_(bias_multi))

    def to(self, device: torch.device):
        self.mlm_model.to(device)
        self.emb_model.to(device)

    def parameters(self, recurse: bool = True):
        yield from self.mlm_model.roberta.parameters()
        yield from self.mlm_model.lm_head.parameters()

    def train(self, mode: bool = True):
        self.mlm_model.train()
        return self

    def eval(self):
        self.mlm_model.eval()
        return self

    def forward(self, origin, masked):
        mask_token_index = torch.where(masked == self.tokenizer.mask_token_id)[1]

        masked_sentence_logits = self.mlm_model(masked).logits
        mask_token_logits = masked_sentence_logits[0, mask_token_index, :]
        one_hot = func.gumbel_softmax(mask_token_logits, hard=True, tau=1000)

        input_embeddings = self.emb_model.embeddings.word_embeddings.weight[masked]
        input_embeddings[masked == self.tokenizer.mask_token_id] = one_hot @ self.emb_model.embeddings.word_embeddings.weight

        embeddings_1 = self.emb_model.forward(inputs_embeds=input_embeddings).pooler_output
        embeddings_2 = self.emb_model(origin).pooler_output

        normalized_embeddings_1 = func.normalize(embeddings_1, p=2)
        normalized_embeddings_2 = func.normalize(embeddings_2, p=2)

        return -torch.mean(normalized_embeddings_1 * normalized_embeddings_2)
