import torch.nn as nn
from transformers import (
    BertModel,
    BertPreTrainedModel,
)


class JointBERT(BertPreTrainedModel):

    def __init__(self, config, wandb_config: dict):
        super().__init__(config)

        self.model = BertModel(config=config)

        self.num_intent_labels = wandb_config['num_intent_labels']
        self.num_slot_labels = wandb_config['num_slot_labels']

        self.intent_classifier = Classifier(config.hidden_size, self.num_intent_labels, wandb_config['dropout'])
        self.slot_classifier = Classifier(config.hidden_size, self.num_slot_labels, wandb_config['dropout'])

        self.intent_loss = nn.CrossEntropyLoss()
        self.slot_loss = nn.CrossEntropyLoss(ignore_index=wandb_config['ignore_index'])

        self.slot_coef = wandb_config['slot_coef']

    def forward(self, input_ids, attention_mask, intent_label_ids, slot_labels_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

        if intent_label_ids is not None:
            total_loss += self.intent_loss(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

        if slot_labels_ids is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = self.slot_loss(active_logits, active_labels)
            else:
                slot_loss = self.slot_loss(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += slot_loss * self.slot_coef

        return total_loss, intent_logits, slot_logits


class Classifier(nn.Module):

    def __init__(self, input_dim, num_labels, dropout=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(self.dropout(x))
