import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    XLMRobertaModel,
)


class XLMRoberta(nn.Module):

    def __init__(self, config, wandb_config: dict):
        super(XLMRoberta, self).__init__()

        self.model = XLMRobertaModel(config=config)

        self.num_intent_labels = wandb_config['num_intent_labels']
        self.num_slot_labels = wandb_config['num_slot_labels']

        self.intent_classifier = Classifier(config.hidden_size, self.num_intent_labels, wandb_config['dropout'])
        self.slot_classifier = Classifier(config.hidden_size, self.num_slot_labels, wandb_config['dropout'])

        self.intent_loss = nn.CrossEntropyLoss()
        self.slot_loss = nn.CrossEntropyLoss(ignore_index=wandb_config['ignore_index'])

        self.slot_coef = wandb_config['slot_coef']

    def forward(self, input_ids, attention_mask, intent_label_ids, slot_labels_ids):
        with autocast():
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


class Classifier(nn.Module):

    def __init__(self, input_dim, num_labels, dropout=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(self.dropout(x))
