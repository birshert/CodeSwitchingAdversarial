import os

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizerFast
from transformers import XLMRobertaConfig
from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizerFast


mapping = {
    'xlm-r': (XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizerFast),
    'm-bert': (BertModel, BertConfig, BertTokenizerFast),
}


class BaseJointModel(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.num_intent_labels = config['num_intent_labels']
        self.num_slot_labels = config['num_slot_labels']

        model_class, config_class, tokenizer_class = mapping[self.__model_name__]

        if config['load_pretrained']:
            self.model = model_class.from_pretrained(self.__parent_name__)
            self.tokenizer = tokenizer_class.from_pretrained(self.__parent_name__)
        else:
            if os.path.exists(self.__cache_path__):
                model_config = config_class.from_json_file(self.__cache_path__ + 'config.json')
            else:
                model_config = config_class.from_pretrained(self.__parent_name__)

            self.model = model_class(config=model_config)
            self.tokenizer = tokenizer_class.from_pretrained(self.__cache_path__)

        self.intent_classifier = Classifier(self.model.config.hidden_size, self.num_intent_labels, config['dropout'])
        self.slot_classifier = Classifier(self.model.config.hidden_size, self.num_slot_labels, config['dropout'])

        if config['load_checkpoint']:
            self.load()

        self.intent_loss = nn.CrossEntropyLoss()
        self.slot_loss = nn.CrossEntropyLoss(ignore_index=config['ignore_index'])

        self.slot_coef = config['slot_coef']

    @property
    def __cache_path__(self):
        return f'models/joint_{self.__model_name__}/'

    @property
    def __model_name__(self):
        raise NotImplementedError

    @property
    def __parent_name__(self):
        raise NotImplementedError

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
        if not os.path.exists(self.__cache_path__):
            os.mkdir(self.__cache_path__)

        torch.save(self.state_dict(), self.__cache_path__ + 'model.pt')
        self.model.config.save_pretrained(self.__cache_path__)

        self.tokenizer.save_pretrained(self.__cache_path__)

    def load(self):
        if not os.path.exists(self.__cache_path__):
            raise OSError('Path does not exist, model cannot be loaded')

        self.load_state_dict(torch.load(self.__cache_path__ + 'model.pt'))

    def load_body(self):
        cache_path = f'models/mlm_{self.__model_name__}/'
        if not os.path.exists(cache_path):
            raise OSError('Path does not exist, model cannot be loaded')

        self.model.load_state_dict(torch.load(cache_path + 'body.pt'))


class JointXLMRoberta(BaseJointModel):

    def __init__(self, config: dict):
        super().__init__(config)

    @property
    def __model_name__(self):
        return 'xlm-r'

    @property
    def __parent_name__(self):
        return 'xlm-roberta-base'


class JointMBERT(BaseJointModel):

    def __init__(self, config: dict):
        super().__init__(config)

    @property
    def __model_name__(self):
        return 'm-bert'

    @property
    def __parent_name__(self):
        return 'bert-base-multilingual-cased'


class Classifier(nn.Module):

    def __init__(self, input_dim, num_labels, dropout=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.linear(self.dropout(x))


class BaseMLMModel(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        model_class, config_class, tokenizer_class = mapping[self.__model_name__]

        if config['load_pretrained']:
            self.model = model_class.from_pretrained(self.__parent_name__)
            self.tokenizer = tokenizer_class.from_pretrained(self.__parent_name__)
        else:
            if os.path.exists(self.__cache_path__):
                model_config = config_class.from_json_file(self.__cache_path__ + 'config.json')
            else:
                model_config = config_class.from_pretrained(self.__parent_name__)

            self.model = model_class(config=model_config)
            self.tokenizer = tokenizer_class.from_pretrained(self.__cache_path__)

        self.lm_head = LMHead(self.model.config)

        if config['load_checkpoint']:
            self.load()

        self.loss = nn.CrossEntropyLoss()

    @property
    def __cache_path__(self):
        return f'models/mlm_{self.__model_name__}/'

    @property
    def __model_name__(self):
        raise NotImplementedError

    @property
    def __parent_name__(self):
        raise NotImplementedError

    def forward(self, input_ids, labels, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)

        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = self.loss(shifted_prediction_scores.view(-1, self.model.config.vocab_size), labels.view(-1))

        return lm_loss, prediction_scores

    def save(self):
        if not os.path.exists(self.__cache_path__):
            os.mkdir(self.__cache_path__)

        torch.save(self.state_dict(), self.__cache_path__ + 'model.pt')
        self.model.config.save_pretrained(self.__cache_path__)

        self.tokenizer.save_pretrained(self.__cache_path__)

    def load(self):
        if not os.path.exists(self.__cache_path__):
            raise OSError('Path does not exist, model cannot be loaded')

        self.load_state_dict(torch.load(self.__cache_path__ + 'model.pt'))

    def save_body(self):
        cache_path = f'models/mlm_{self.__model_name__}/'
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        torch.save(self.model.state_dict(), cache_path + 'body.pt')


class MLMXLMRoberta(BaseMLMModel):

    def __init__(self, config: dict):
        super().__init__(config)

    @property
    def __model_name__(self):
        return 'xlm-r'

    @property
    def __parent_name__(self):
        return 'xlm-roberta-base'


class MLMMBERT(BaseMLMModel):

    def __init__(self, config: dict):
        super().__init__(config)

    @property
    def __model_name__(self):
        return 'm-bert'

    @property
    def __parent_name__(self):
        return 'bert-base-multilingual-cased'


class LMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gelu = nn.GELU()

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x
