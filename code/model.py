import os

import torch
import torch.nn as nn
from transformers import (
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)


class XLMRoberta(nn.Module):

    def __init__(self, config: dict):
        super(XLMRoberta, self).__init__()

        self.num_intent_labels = config['num_intent_labels']
        self.num_slot_labels = config['num_slot_labels']

        if config['load_pretrained']:
            self.model = XLMRobertaModel.from_pretrained(self.__parent_name__)
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.__parent_name__)
        else:
            if os.path.exists(f'models/{self.__model_name__}'):
                model_config = XLMRobertaConfig.from_json_file(f'models/{self.__model_name__}/config.json')
            else:
                model_config = XLMRobertaConfig.from_pretrained(self.__parent_name__)

            self.model = XLMRobertaModel(config=model_config)
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(f'models/{self.__model_name__}/')

        self.intent_classifier = Classifier(self.model.config.hidden_size, self.num_intent_labels, config['dropout'])
        self.slot_classifier = Classifier(self.model.config.hidden_size, self.num_slot_labels, config['dropout'])

        if config['load_checkpoint']:
            self.load()

        self.intent_loss = nn.CrossEntropyLoss()
        self.slot_loss = nn.CrossEntropyLoss(ignore_index=config['ignore_index'])

        self.slot_coef = config['slot_coef']

    @property
    def __model_name__(self):
        return 'xlm-r'

    @property
    def __parent_name__(self):
        return 'xlm-roberta-base'

    def forward(self, input_ids, intent_label_ids, slot_labels_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        intent_loss = self.intent_loss(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

        active_loss = attention_mask.view(-1) == 1
        active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
        active_labels = slot_labels_ids.view(-1)[active_loss]
        slot_loss = self.slot_loss(active_logits, active_labels)

        return intent_loss + slot_loss * self.slot_coef, intent_logits, slot_logits

    def save(self):
        if not os.path.exists(f'models/{self.__model_name__}'):
            os.mkdir(f'models/{self.__model_name__}')

        torch.save(self.state_dict(), f'models/{self.__model_name__}/model.pt')
        self.model.config.save_pretrained(f'models/{self.__model_name__}/')

        self.tokenizer.save_pretrained(f'models/{self.__model_name__}/')

    def load(self):
        if not os.path.exists(f'models/{self.__model_name__}'):
            raise OSError('Path does not exist, model cannot be loaded')

        self.load_state_dict(torch.load(f'models/{self.__model_name__}/model.pt'))


class Classifier(nn.Module):

    def __init__(self, input_dim, num_labels, dropout=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(self.dropout(x))
